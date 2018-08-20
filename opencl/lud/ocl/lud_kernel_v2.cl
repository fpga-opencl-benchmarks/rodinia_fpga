#include "../common/common.h"

// diameter kernel settings
#ifndef DIA_UNROLL
	#define DIA_UNROLL	1 // nearly unusable since increases number of writes to local buffers to over 3, resulting in stallable accesses
#endif
#define DIA_SIMD		1 // unusable due to thread-id-dependent branching
#define DIA_CU			1 // this kernel is only run by one work-group; hence, no point in compute unit replication

// perimeter kernel settings
#ifndef PERI_UNROLL
	#define PERI_UNROLL	1 // nearly unusable since increases number of writes to local buffers to over 3, resulting in stallable accesses
#endif
#define PERI_SIMD		1 // unusable due to thread-id-dependent branching
#ifndef PERI_CU
	#define PERI_CU	4
#endif

// internal kernel settings
#ifndef INT_SIMD
	#define INT_SIMD	2
#endif
#ifndef INT_CU
	#define INT_CU		1
#endif

__attribute__((num_simd_work_items(DIA_SIMD)))
__attribute__((num_compute_units(DIA_CU)))
__attribute__((reqd_work_group_size(BSIZE,1,1)))
__kernel void lud_diagonal(__global float* restrict m,
                                    int             matrix_dim,
                                    int             offset)
{ 
	int i,j;
	int tx = get_local_id(0);

	__local float shadow[BSIZE * BSIZE];

	int array_offset = offset*matrix_dim+offset;
	for(i=0; i < BSIZE; i++)
	{
		shadow[i * BSIZE + tx]=m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(i=0; i < BSIZE-1; i++)
	{
		if (tx>i)
		{
			#pragma unroll DIA_UNROLL
			for(j=0; j < i; j++)
			{
				shadow[tx * BSIZE + i] -= shadow[tx * BSIZE + j] * shadow[j * BSIZE + i];
			}
			shadow[tx * BSIZE + i] /= shadow[i * BSIZE + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx>i)
		{
			#pragma unroll DIA_UNROLL
			for(j=0; j < i+1; j++)
			{
				shadow[(i+1) * BSIZE + tx] -= shadow[(i+1) * BSIZE + j]*shadow[j * BSIZE + tx];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	array_offset = (offset+1)*matrix_dim+offset;
	for(i=1; i < BSIZE; i++)
	{
		m[array_offset+tx]=shadow[i * BSIZE + tx];
		array_offset += matrix_dim;
	}
}

__attribute__((num_compute_units(PERI_CU)))
__attribute__((num_simd_work_items(PERI_SIMD)))
__attribute__((reqd_work_group_size(2 * BSIZE,1,1)))
__kernel void lud_perimeter(__global float* restrict m,
                                     int             matrix_dim,
                                     int             offset)
{
	int i,j, array_offset;
	int idx;

	int  bx = get_group_id(0);	
	int  tx = get_local_id(0);

	__local float dia[BSIZE * BSIZE], peri_col[BSIZE * BSIZE], peri_row[BSIZE * BSIZE];

	if (tx < BSIZE)
	{
		idx = tx;
		#pragma unroll 1 //to disable auto unroll which results in explosion of ports to external memory
		array_offset = offset*matrix_dim+offset;
		for (i=0; i < BSIZE/2; i++)
		{
			dia[i * BSIZE + idx]=m[array_offset+idx];
			array_offset += matrix_dim;
		}
    
		array_offset = offset*matrix_dim+offset;
		for (i=0; i < BSIZE; i++)
		{
			peri_row[i * BSIZE + idx]=m[array_offset+(bx+1)*BSIZE+idx];
			array_offset += matrix_dim;
		}
	}
	else
	{
		idx = tx-BSIZE;
		array_offset = (offset+BSIZE/2)*matrix_dim+offset;
		#pragma unroll 1 //to disable auto unroll which results in explosion of ports to external memory
		for (i=BSIZE/2; i < BSIZE; i++)
		{
			dia[i * BSIZE + idx]=m[array_offset+idx];
			array_offset += matrix_dim;
		}
	
		array_offset = (offset+(bx+1)*BSIZE)*matrix_dim+offset;
		for (i=0; i < BSIZE; i++)
		{
			peri_col[i * BSIZE + idx] = m[array_offset+idx];
			array_offset += matrix_dim;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (tx < BSIZE)
	{ //peri-row
		idx=tx;
		for(i=1; i < BSIZE; i++)
		{
			#pragma unroll PERI_UNROLL
			for (j=0; j < i; j++)
			{
				peri_row[i * BSIZE + idx]-=dia[i * BSIZE + j]*peri_row[j * BSIZE + idx];
			}
		}
	}
	else
	{ //peri-col
		idx=tx - BSIZE;
		for(i=0; i < BSIZE; i++)
		{
			#pragma unroll PERI_UNROLL
			for(j=0; j < i; j++)
			{
				peri_col[idx * BSIZE + i]-=dia[j * BSIZE + i]*peri_col[idx * BSIZE + j];
			}
			peri_col[idx * BSIZE + i] /= dia[i * BSIZE+ i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    
	if (tx < BSIZE)
	{ //peri-row
		idx=tx;
		array_offset = (offset+1)*matrix_dim+offset;
		for(i=1; i < BSIZE; i++)
		{
			m[array_offset+(bx+1)*BSIZE+idx] = peri_row[i*BSIZE+idx];
			array_offset += matrix_dim;
		}
	}
	else
	{ //peri-col
		idx=tx - BSIZE;
		array_offset = (offset+(bx+1)*BSIZE)*matrix_dim+offset;
		for(i=0; i < BSIZE; i++)
		{
			m[array_offset+idx] = peri_col[i*BSIZE+idx];
			array_offset += matrix_dim;
		}
	}
}

__attribute__((num_compute_units(INT_CU)))
__attribute__((num_simd_work_items(INT_SIMD)))
__attribute__((reqd_work_group_size(BSIZE,BSIZE,1)))
__kernel void lud_internal(__global float* restrict m,
                                    int             matrix_dim,
                                    int             offset)
{
	int  bx = get_group_id(0);
	int  by = get_group_id(1);
  
	int  tx = get_local_id(0);
	int  ty = get_local_id(1);

	__local float peri_col[BSIZE * BSIZE], peri_row[BSIZE * BSIZE];

	int i;
	float sum;

	int global_row_id = offset + (by+1)*BSIZE;
	int global_col_id = offset + (bx+1)*BSIZE;

	peri_row[ty * BSIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
	peri_col[ty * BSIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	sum = 0;
	#pragma unroll
	for (i=0; i < BSIZE; i++)
	{
		sum += peri_col[ty * BSIZE + i] * peri_row[i * BSIZE + tx];
	}
	m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;
}
