#ifndef BLOCK_SIZE
	#define BLOCK_SIZE 64
#endif

#define DIA_UNROLL    2
#define PERI_UNROLL   8
#define PERI_SIMD     1
#define PERI_COMPUTE  2
#define INTER_SIMD    1
#define INTER_COMPUTE 3

#define LMEM_SIZE BLOCK_SIZE*BLOCK_SIZE
#define DIA_LMEM_ATTRIB __attribute__((memory, numbanks(2), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1)))
//#define PERI_LMEM_ATTRIB1 __attribute__((memory, numbanks(1), bankwidth(8*PERI_UNROLL)))
//#define PERI_LMEM_ATTRIB2 __attribute__((memory, numbanks(2), bankwidth(8*PERI_UNROLL)))

#include "../common/opencl_kernel_common.h"

__attribute__((reqd_work_group_size(BLOCK_SIZE,1,1)))
__kernel void lud_diagonal(__global volatile float* RESTRICT m, 
                                             int             matrix_dim,
                                             int             offset)
{ 
	int i,j;
	int tx = get_local_id(0);
	__local float DIA_LMEM_ATTRIB shadow_row[LMEM_SIZE], DIA_LMEM_ATTRIB shadow_col[LMEM_SIZE];

	int array_offset = offset * matrix_dim + offset;
	for(i=0; i < BLOCK_SIZE; i++)
	{
		shadow_row[i * BLOCK_SIZE + tx] = m[array_offset + tx];
		shadow_col[tx * BLOCK_SIZE + i] = m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(i=0; i < BLOCK_SIZE-1; i++)
	{
		if (tx>i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(j=0; j < i; j++)
			{
				sum += shadow_row[tx * BLOCK_SIZE + j] * shadow_col[i * BLOCK_SIZE + j];
			}
			shadow_row[tx * BLOCK_SIZE + i] = (shadow_row[tx * BLOCK_SIZE + i] - sum) / shadow_col[i * BLOCK_SIZE + i];
			shadow_col[i * BLOCK_SIZE + tx] = shadow_row[tx * BLOCK_SIZE + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx>i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(j=0; j < i+1; j++)
			{
				sum += shadow_row[(i+1) * BLOCK_SIZE + j] * shadow_col[tx * BLOCK_SIZE + j];
			}
			shadow_row[(i+1) * BLOCK_SIZE + tx] -= sum;
			shadow_col[tx * BLOCK_SIZE + (i+1)] = shadow_row[(i+1) * BLOCK_SIZE + tx];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	array_offset = (offset+1) * matrix_dim + offset;
	for(i=1; i < BLOCK_SIZE; i++)
	{
		m[array_offset + tx] = shadow_row[i * BLOCK_SIZE + tx];
		array_offset += matrix_dim;
	}
}

__attribute__((num_compute_units(PERI_COMPUTE)))
__attribute__((num_simd_work_items(PERI_SIMD)))
__attribute__((max_work_group_size(2*BLOCK_SIZE))) // should be reqd, used max instead to force the compiler into reducing the number of simultaneous work groups and over-replicating the local buffers
__kernel void lud_perimeter_row(__global volatile float* RESTRICT m,
                                                  int             matrix_dim,
                                                  int             offset)
{
	int i, j, array_offset1, array_offset2;
	__local float dia_row[LMEM_SIZE], peri_row[LMEM_SIZE];

	int  bx = get_group_id(0);
	int  tx = get_local_id(0);

	array_offset1 = offset * matrix_dim + offset;
	array_offset2 = offset * matrix_dim + (bx+1) * BLOCK_SIZE + offset;
	for (i=0; i < BLOCK_SIZE; i++)
	{
		dia_row[i * BLOCK_SIZE + tx] = m[array_offset1 + tx];
		peri_row[tx * BLOCK_SIZE + i] = m[array_offset2 + tx];

		array_offset1 += matrix_dim;
		array_offset2 += matrix_dim;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	array_offset1 = offset * matrix_dim + (bx+1) * BLOCK_SIZE + offset;
	for(i=0; i < BLOCK_SIZE; i++)
	{
		float sum = 0.0f;
		#pragma unroll PERI_UNROLL
		for (j=0; j < i; j++)
		{
			sum += dia_row[i * BLOCK_SIZE + j] * peri_row[tx * BLOCK_SIZE + j];
		}
		peri_row[tx * BLOCK_SIZE + i] -= sum;

		m[array_offset1 + tx] = peri_row[tx * BLOCK_SIZE + i];
		array_offset1 += matrix_dim;
	}
}

__attribute__((num_compute_units(PERI_COMPUTE)))
__attribute__((num_simd_work_items(PERI_SIMD)))
__attribute__((max_work_group_size(2*BLOCK_SIZE))) // should be reqd, used max instead to force the compiler into reducing the number of simultaneous work groups and over-replicating the local buffers
__kernel void lud_perimeter_col(__global volatile float* RESTRICT m,
                                                  int             matrix_dim,
                                                  int             offset)
{
	int i, j, array_offset1, array_offset2;
	__local float dia_col[LMEM_SIZE], peri_col[LMEM_SIZE];

	int  bx = get_group_id(0);	
	int  tx = get_local_id(0);

	array_offset1 = offset * matrix_dim + offset;
	array_offset2 = (offset + (bx+1) * BLOCK_SIZE) * matrix_dim + offset;
	for (i=0; i < BLOCK_SIZE; i++)
	{
		dia_col[tx * BLOCK_SIZE + i] = m[array_offset1 + tx];
		peri_col[i * BLOCK_SIZE + tx] = m[array_offset2 + tx];

		array_offset1 += matrix_dim;
		array_offset2 += matrix_dim;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(i=0; i < BLOCK_SIZE; i++)
	{
		float sum = 0.0f;
		#pragma unroll PERI_UNROLL
		for(j=0; j < i; j++)
		{
			sum += dia_col[i * BLOCK_SIZE + j] * peri_col[tx * BLOCK_SIZE + j];
		}
		peri_col[tx * BLOCK_SIZE + i] = (peri_col[tx * BLOCK_SIZE + i] - sum) / dia_col[i * BLOCK_SIZE + i];
	}

	//barrier(CLK_LOCAL_MEM_FENCE);

	array_offset1 = (offset + (bx+1) * BLOCK_SIZE) * matrix_dim + offset;
	#pragma unroll 16
	for(i=0; i < BLOCK_SIZE; i++)
	{
		m[array_offset1 + tx * matrix_dim + i] = peri_col[tx * BLOCK_SIZE + i];
	}
}

__attribute__((num_compute_units(INTER_COMPUTE)))
__attribute__((num_simd_work_items(INTER_SIMD)))
__attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__kernel void lud_internal(__global float* RESTRICT m,
                                    int             matrix_dim,
                                    int             offset)
{
	int i;
	float sum;
	__local float peri_row[LMEM_SIZE], peri_col[LMEM_SIZE];

	int  bx = get_group_id(0);
	int  by = get_group_id(1);
  
	int  tx = get_local_id(0);
	int  ty = get_local_id(1);

	int global_row_id = offset + (by+1) * BLOCK_SIZE;
	int global_col_id = offset + (bx+1) * BLOCK_SIZE;

	peri_row[ty * BLOCK_SIZE + tx] = m[(offset + ty) * matrix_dim + global_col_id + tx];
	peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id + ty) * matrix_dim + offset + tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	sum = 0;
	#pragma unroll
	for (i=0; i < BLOCK_SIZE; i++)
	{
		sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
	}
	m[(global_row_id + ty) * matrix_dim + global_col_id + tx] -= sum;
}
