#ifndef BLOCK_SIZE
	#ifdef AOCL_BOARD_a10pl4_gx115es3
		#define BLOCK_SIZE 128
	#else
		#define BLOCK_SIZE 64
	#endif
#endif

#ifdef AOCL_BOARD_a10pl4_gx115es3
	#define DIA_UNROLL    4
	#define PERI_UNROLL   16
	#define PERI_SIMD     1
	#define PERI_COMPUTE  2
	#define INTER_SIMD    4
	#define INTER_COMPUTE 1
#else
	#define DIA_UNROLL    2
	#define PERI_UNROLL   8
	#define PERI_SIMD     1
	#define PERI_COMPUTE  2
	#define INTER_SIMD    1
	#define INTER_COMPUTE 3
#endif

// Enable volatile for Arria 10 in the internal kernel
#ifdef AOCL_BOARD_a10pl4_gx115es3
	#define VOLATILE volatile
#else
	#define VOLATILE
#endif

#define LMEM_SIZE BLOCK_SIZE*BLOCK_SIZE
#define DIA_LMEM_ATTRIB __attribute__((memory, numbanks(2), bankwidth(4*DIA_UNROLL), doublepump, numreadports(3), numwriteports(1)))

#include "../common/opencl_kernel_common.h"

// The Arria 10 version uses the old diameter implementation without splitting the shadow buffer since that optimization only reduces Block RAM usage
// when there are enough Block RAMs to enable stall-free access. On Arria 10, there aren't enough Block RAMs either way, and accesses are stallable anyway,
// using two buffers will actually increase memory usage in this case
#ifdef AOCL_BOARD_a10pl4_gx115es3
__attribute__((reqd_work_group_size(BLOCK_SIZE,1,1)))
__kernel void lud_diagonal(__global volatile float* RESTRICT m, 
                                             int             matrix_dim,
                                             int             offset)
{ 
	int i,j;
	int tx = get_local_id(0);
	__local float shadow[LMEM_SIZE];

	int array_offset = offset * matrix_dim + offset;
	for(i=0; i < BLOCK_SIZE; i++)
	{
		shadow[i * BLOCK_SIZE + tx] = m[array_offset + tx];
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
				sum += shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
			}
			shadow[tx * BLOCK_SIZE + i] = (shadow[tx * BLOCK_SIZE + i] - sum) / shadow[i * BLOCK_SIZE + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx>i)
		{
			float sum = 0.0f;
			#pragma unroll DIA_UNROLL
			for(j=0; j < i+1; j++)
			{
				sum += shadow[(i+1) * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + tx];
			}
			shadow[(i+1) * BLOCK_SIZE + tx] -= sum;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	array_offset = (offset+1) * matrix_dim + offset;
	for(i=1; i < BLOCK_SIZE; i++)
	{
		m[array_offset + tx] = shadow[i * BLOCK_SIZE + tx];
		array_offset += matrix_dim;
	}
}

#else
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
#endif

__attribute__((num_compute_units(PERI_COMPUTE)))
__attribute__((num_simd_work_items(PERI_SIMD)))
__attribute__((reqd_work_group_size(BLOCK_SIZE*2,1,1)))
__kernel void lud_perimeter(__global volatile float* RESTRICT m,
                                              int             matrix_dim,
                                              int             offset)
{
	int i,j, array_offset;
	int idx;
	__local float dia_row[LMEM_SIZE], dia_col[LMEM_SIZE], peri_row[LMEM_SIZE], peri_col[LMEM_SIZE];

	int  bx = get_group_id(0);	
	int  tx = get_local_id(0);

	if (tx < BLOCK_SIZE)
	{
		idx = tx;
		array_offset = offset * matrix_dim + offset;
		for (i=0; i < BLOCK_SIZE/2; i++)
		{
			dia_row[i * BLOCK_SIZE + idx] = m[array_offset + idx];
			dia_col[idx * BLOCK_SIZE + i] = m[array_offset + idx];
			array_offset += matrix_dim;
		}
    
		array_offset = offset * matrix_dim + (bx+1) * BLOCK_SIZE + offset;
		for (i=0; i < BLOCK_SIZE; i++)
		{
			peri_row[idx * BLOCK_SIZE + i] = m[array_offset + idx];
			array_offset += matrix_dim;
		}
	}
	else
	{
		idx = tx - BLOCK_SIZE;
		array_offset = (offset + BLOCK_SIZE/2) * matrix_dim + offset;
		for (i=BLOCK_SIZE/2; i < BLOCK_SIZE; i++)
		{
			dia_row[i * BLOCK_SIZE + idx] = m[array_offset + idx];
			dia_col[idx * BLOCK_SIZE + i] = m[array_offset + idx];
			array_offset += matrix_dim;
		}
	
		array_offset = (offset + (bx+1) * BLOCK_SIZE) * matrix_dim + offset;
		for (i=0; i < BLOCK_SIZE; i++)
		{
			peri_col[i * BLOCK_SIZE + idx] = m[array_offset + idx];
			array_offset += matrix_dim;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (tx < BLOCK_SIZE)
	{ //peri-row
		idx = tx;
		for(i=1; i < BLOCK_SIZE; i++)
		{
			float sum = 0.0f;
			#pragma unroll PERI_UNROLL
			for (j=0; j < i; j++)
			{
				sum += dia_row[i * BLOCK_SIZE + j] * peri_row[idx * BLOCK_SIZE + j];
			}
			peri_row[idx * BLOCK_SIZE + i] -= sum;
		}
	}
	else
	{ //peri-col
		idx = tx - BLOCK_SIZE;
		for(i=0; i < BLOCK_SIZE; i++)
		{
			float sum = 0.0f;
			#pragma unroll PERI_UNROLL
			for(j=0; j < i; j++)
			{
				sum += dia_col[i * BLOCK_SIZE + j] * peri_col[idx * BLOCK_SIZE + j];
			}
			peri_col[idx * BLOCK_SIZE + i] = (peri_col[idx * BLOCK_SIZE + i] - sum) / dia_col[i * BLOCK_SIZE + i];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
    
	if (tx < BLOCK_SIZE)
	{ //peri-row
		idx = tx;
		array_offset = (offset+1) * matrix_dim + (bx+1) * BLOCK_SIZE + offset;
		for(i=1; i < BLOCK_SIZE; i++)
		{
			m[array_offset + idx] = peri_row[idx * BLOCK_SIZE + i];
			array_offset += matrix_dim;
		}
	}
	else
	{ //peri-col
		idx = tx - BLOCK_SIZE;
		array_offset = (offset + (bx+1) * BLOCK_SIZE) * matrix_dim + offset;
		for(i=0; i < BLOCK_SIZE; i++)
		{
			m[array_offset + idx] =  peri_col[i * BLOCK_SIZE + idx];
			array_offset += matrix_dim;
		}
	}
}

__attribute__((num_compute_units(INTER_COMPUTE)))
__attribute__((num_simd_work_items(INTER_SIMD)))
__attribute__((reqd_work_group_size(BLOCK_SIZE,BLOCK_SIZE,1)))
// The Arria 10 version uses SIMD instead of multiple compute units for this kernel and the private cache seems to have a positive effect
// only if multiple compute units are used. Because of this, for the Arria 10 version, volatile is used to disable the cache and save Block RAMs
__kernel void lud_internal(__global VOLATILE float* RESTRICT m,
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

// The following lines reverse the direction of loading peri_row for Arria 10 similar to what is done in the previous kernels
// Originally, coalescing accesses in this buffer in the unrolled loop below should not have been possible since accesses to that buffer
// based on i are not consecutive, but for some reason, probably due to full unrolling, this didn't cause an issue.
// Because of this, reversing the access should not have any effect either, but Altera's report for the Arria 10 version claimed that
// doing this saves a few Block RAMs, which, after placement and routing, was proven to be false.
// Still, doing this results in higher operating frequency, not because of the optimization, but because of luck
#ifdef AOCL_BOARD_a10pl4_gx115es3
	peri_row[tx * BLOCK_SIZE + ty] = m[(offset + ty) * matrix_dim + global_col_id + tx];
#else
	peri_row[ty * BLOCK_SIZE + tx] = m[(offset + ty) * matrix_dim + global_col_id + tx];
#endif

	peri_col[ty * BLOCK_SIZE + tx] = m[(global_row_id + ty) * matrix_dim + offset + tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	sum = 0;
	#pragma unroll
	for (i=0; i < BLOCK_SIZE; i++)
	{
#ifdef AOCL_BOARD_a10pl4_gx115es3
		sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[tx * BLOCK_SIZE + i];
#else
		sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
#endif
	}
	m[(global_row_id + ty) * matrix_dim + global_col_id + tx] -= sum;
}
