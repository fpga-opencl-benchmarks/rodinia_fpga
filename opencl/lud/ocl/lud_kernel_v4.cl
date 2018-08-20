#include "../common/common.h"

#ifndef DIA_UNROLL
	#ifdef AOCL_BOARD_de5net_a7
		#define DIA_UNROLL	4
	#else
		#define DIA_UNROLL	2
	#endif
#endif
#ifndef PERI_UNROLL
	#ifdef AOCL_BOARD_de5net_a7
		#define PERI_UNROLL	8
	#else
		#define PERI_UNROLL	4		
	#endif
#endif
#ifndef PERI_CU
	#ifdef AOCL_BOARD_de5net_a7
		#define PERI_CU	2
	#else
		#define PERI_CU	2
	#endif
#endif
#ifndef INT_SIMD
	#ifdef AOCL_BOARD_de5net_a7
		#define INT_SIMD	2
	#else
		#define INT_SIMD	1
	#endif
#endif
#ifndef INT_CU
	#ifdef AOCL_BOARD_de5net_a7
		#define INT_CU		1
	#else
		#define INT_CU		1
	#endif
#endif

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__kernel void lud_diagonal(__global float* restrict m, 
                                    int             matrix_dim,
                                    int             offset)
{ 
	int tx = get_local_id(0);
	__local float __attribute__((memory, numbanks(1), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1))) shadow_row[BSIZE * BSIZE];
	__local float __attribute__((memory, numbanks(1), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1))) shadow_col[BSIZE * BSIZE];

	int array_offset = offset;
	for(int i = 0; i < BSIZE; i++)
	{
		shadow_row[i * BSIZE + tx] = m[array_offset + tx];
		shadow_col[tx * BSIZE + i] = m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);

	array_offset = offset + matrix_dim;
	for(int i = 0; i < BSIZE - 1; i++)
	{
		if (tx > i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(int j = 0; j < i; j++)
			{
				sum += shadow_row[tx * BSIZE + j] * shadow_col[i * BSIZE + j];
			}
			shadow_row[tx * BSIZE + i] = (shadow_row[tx * BSIZE + i] - sum) / shadow_col[i * BSIZE + i];
			//shadow_col[i * BSIZE + tx] = shadow_row[tx * BSIZE + i];		// commented out since it is not actually required and output is correct either way
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx > i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(int j = 0; j < i + 1; j++)
			{
				sum += shadow_row[(i + 1) * BSIZE + j] * shadow_col[tx * BSIZE + j];
			}
			shadow_row[(i + 1) * BSIZE + tx] -= sum;
			shadow_col[tx * BSIZE + (i + 1)] = shadow_row[(i + 1) * BSIZE + tx];
		}
		m[array_offset + tx] = shadow_row[(i + 1) * BSIZE + tx];
		array_offset += matrix_dim;
	}
}

__attribute__((num_compute_units(PERI_CU)))
__attribute__((reqd_work_group_size(BSIZE * 2,1,1)))
__kernel void lud_perimeter(__global float* restrict m,
                                     int             matrix_dim,
                                     int             offset)
{
	__local float dia_row[BSIZE * BSIZE], dia_col[BSIZE * BSIZE], peri_row[BSIZE * BSIZE];
	__local float __attribute__((memory, numbanks(1), bankwidth(4*PERI_UNROLL), doublepump, numreadports(3), numwriteports(2))) peri_col[BSIZE * BSIZE];

	int bx = get_group_id(0);
	int tx = get_local_id(0);

	int idx = tx % BSIZE;
	int txg = tx / BSIZE;

	int constant_1 = txg * matrix_dim;
	int constant_2 = (bx + 1) * BSIZE;
	int constant_3 = (bx + 1) * BSIZE * matrix_dim;

	int array_offset_1 = offset + constant_1;
	int array_offset_2 = offset + constant_1 + constant_2;
	int array_offset_3 = offset + constant_1 + constant_3;

	// two block rows are read per iteration
	for (int i = 0; i < BSIZE; i = i + 2)
	{
		dia_row[(i + txg) * BSIZE + idx]  = m[array_offset_1 + idx];
		dia_col[idx * BSIZE + (i + txg)]  = m[array_offset_1 + idx];
		peri_row[idx * BSIZE + (i + txg)] = m[array_offset_2 + idx];
		peri_col[(i + txg) * BSIZE + idx] = m[array_offset_3 + idx];

		array_offset_1 += 2 * matrix_dim;
		array_offset_2 += 2 * matrix_dim;
		array_offset_3 += 2 * matrix_dim;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tx < BSIZE)
	{ //peri-row
		int idx = tx;
		int peri_row_array_offset = offset + constant_2;
		for(int i = 0; i < BSIZE; i++)
		{
			float sum = 0.0f;
			#pragma unroll PERI_UNROLL
			for (int j = 0; j < i; j++)
			{
				sum += dia_row[i * BSIZE + j] * peri_row[idx * BSIZE + j];
			}
			peri_row[idx * BSIZE + i] -= sum;

			// write-back is done here since it removes one extra read from the peri_row buffer
			// and accesses to external memory are consecutive based on work-group ID anyway
			m[peri_row_array_offset + idx] = peri_row[idx * BSIZE + i];
			peri_row_array_offset += matrix_dim;
		}
	}
	else
	{ //peri-col
		int idx = tx - BSIZE;
		for(int i = 0; i < BSIZE; i++)
		{
			float sum = 0.0f;
			#pragma unroll PERI_UNROLL
			for(int j = 0; j < i; j++)
			{
				sum += dia_col[i * BSIZE + j] * peri_col[idx * BSIZE + j];
			}
			peri_col[idx * BSIZE + i] = (peri_col[idx * BSIZE + i] - sum) / dia_col[i * BSIZE + i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	int peri_col_array_offset = offset + constant_1 + constant_3;
	// two block rows are written per iteration, disable compiler auto unrolling
	#pragma unroll 1
	for(int i = 0; i < BSIZE; i = i + 2)
	{
		// even though this could also be merged into the compute loop like the other write-back,
		// it was avoided since it would have resulted in accesses that are not consecutive based
		// on work-group ID and lowered performance
		m[peri_col_array_offset + idx] = peri_col[(i + txg) * BSIZE + idx];
		peri_col_array_offset += 2 * matrix_dim;
	}
}

__attribute__((num_compute_units(INT_CU)))
__attribute__((num_simd_work_items(INT_SIMD)))
__attribute__((reqd_work_group_size(BSIZE,BSIZE,1)))
__kernel void lud_internal(__global float* restrict m,
                                    int             matrix_dim,
                                    int             offset)
{
	__local float peri_row[BSIZE * BSIZE], peri_col[BSIZE * BSIZE];

	int bx = get_group_id(0);
	int by = get_group_id(1);
  
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	int global_row_id = (by + 1) * BSIZE;
	int global_col_id = (bx + 1) * BSIZE;

	peri_row[ty * BSIZE + tx] = m[offset + (ty) * matrix_dim + global_col_id + tx];
	peri_col[ty * BSIZE + tx] = m[offset + (global_row_id + ty) * matrix_dim + tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0;
	#pragma unroll
	for (int i = 0; i < BSIZE; i++)
	{
		sum += peri_col[ty * BSIZE + i] * peri_row[i * BSIZE + tx];
	}
	m[offset + (global_row_id + ty) * matrix_dim + global_col_id + tx] -= sum;
}
