// Setting BLOCK_SIZE
// #define BLOCK_SIZE 16
// BLOCK_SIZE is already defined in host code, and is passed to the
// JIT compiler. problem_size.h is used to set the problem size when
// compiled with an AOT compiler.
#ifdef USE_AOT
#include "problem_size.h"
#endif

#define LMEM_SIZE __attribute__((local_mem_size(BLOCK_SIZE*BLOCK_SIZE*4))) 

#include "../common/opencl_kernel_common.h"

__attribute__((reqd_work_group_size(BLOCK_SIZE,1,1)))
__kernel void 
lud_diagonal(__global float* RESTRICT m, 
LMEM_SIZE    __local  float* RESTRICT shadow,
                      int             matrix_dim, 
                      int             offset)
{ 
	int i,j;
	int tx = get_local_id(0);

	int array_offset = offset*matrix_dim+offset;
	for(i=0; i < BLOCK_SIZE; i++)
	{
		shadow[i * BLOCK_SIZE + tx]=m[array_offset + tx];
		array_offset += matrix_dim;
	}
  
	barrier(CLK_LOCAL_MEM_FENCE);
	
	for(i=0; i < BLOCK_SIZE-1; i++)
	{
		if (tx>i)
		{
			for(j=0; j < i; j++)
			{
				shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
			}
			shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx>i)
		{
			for(j=0; j < i+1; j++)
			{
				shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	array_offset = (offset+1)*matrix_dim+offset;
	for(i=1; i < BLOCK_SIZE; i++)
	{
		m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
		array_offset += matrix_dim;
	}
}

__attribute__((reqd_work_group_size(BLOCK_SIZE*2,1,1)))
__kernel void
lud_perimeter(__global float* RESTRICT m, 
LMEM_SIZE     __local  float* RESTRICT dia,
LMEM_SIZE     __local  float* RESTRICT peri_row,
LMEM_SIZE     __local  float* RESTRICT peri_col,
                       int             matrix_dim, 
                       int             offset)
{
	int i,j, array_offset;
	int idx;

	int  bx = get_group_id(0);	
	int  tx = get_local_id(0);

	if (tx < BLOCK_SIZE)
	{
		idx = tx;
		array_offset = offset*matrix_dim+offset;
		for (i=0; i < BLOCK_SIZE/2; i++)
		{
			dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
			array_offset += matrix_dim;
		}
    
		array_offset = offset*matrix_dim+offset;
		for (i=0; i < BLOCK_SIZE; i++)
		{
			peri_row[i * BLOCK_SIZE + idx]=m[array_offset+(bx+1)*BLOCK_SIZE+idx];
			array_offset += matrix_dim;
		}
	}
	else
	{
		idx = tx-BLOCK_SIZE;
		array_offset = (offset+BLOCK_SIZE/2)*matrix_dim+offset;
		for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++)
		{
			dia[i * BLOCK_SIZE + idx]=m[array_offset+idx];
			array_offset += matrix_dim;
		}
	
		array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
		for (i=0; i < BLOCK_SIZE; i++)
		{
			peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
			array_offset += matrix_dim;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (tx < BLOCK_SIZE)
	{ //peri-row
		idx=tx;
		for(i=1; i < BLOCK_SIZE; i++)
		{
			for (j=0; j < i; j++)
			{
				peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE + j]*peri_row[j * BLOCK_SIZE + idx];
			}
		}
	}
	else
	{ //peri-col
		idx=tx - BLOCK_SIZE;
		for(i=0; i < BLOCK_SIZE; i++)
		{
			for(j=0; j < i; j++)
			{
				peri_col[idx * BLOCK_SIZE + i]-=dia[j * BLOCK_SIZE + i]*peri_col[idx * BLOCK_SIZE + j];
			}
			peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    
	if (tx < BLOCK_SIZE)
	{ //peri-row
		idx=tx;
		array_offset = (offset+1)*matrix_dim+offset;
		for(i=1; i < BLOCK_SIZE; i++)
		{
			m[array_offset+(bx+1)*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
			array_offset += matrix_dim;
		}
	}
	else
	{ //peri-col
		idx=tx - BLOCK_SIZE;
		array_offset = (offset+(bx+1)*BLOCK_SIZE)*matrix_dim+offset;
		for(i=0; i < BLOCK_SIZE; i++)
		{
			m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
			array_offset += matrix_dim;
		}
	}
}

__attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
__kernel void
lud_internal(__global float* RESTRICT m, 
LMEM_SIZE    __local  float* RESTRICT peri_row,
LMEM_SIZE    __local  float* RESTRICT peri_col,
                      int             matrix_dim, 
                      int             offset)
{
	int  bx = get_group_id(0);
	int  by = get_group_id(1);
  
	int  tx = get_local_id(0);
	int  ty = get_local_id(1);

	int i;
	float sum;

	int global_row_id = offset + (by+1)*BLOCK_SIZE;
	int global_col_id = offset + (bx+1)*BLOCK_SIZE;

	peri_row[ty * BLOCK_SIZE + tx] = m[(offset+ty)*matrix_dim+global_col_id+tx];
	peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id+ty)*matrix_dim+offset+tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	sum = 0;
	#pragma unroll
	for (i=0; i < BLOCK_SIZE; i++)
	{
		sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
	}
	m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;
}
