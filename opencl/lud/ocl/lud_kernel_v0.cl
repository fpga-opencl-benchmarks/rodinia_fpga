#include "../common/common.h"

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
			for(j=0; j < i; j++)
			{
				shadow[tx * BSIZE + i] -= shadow[tx * BSIZE + j] * shadow[j * BSIZE + i];
			}
			shadow[tx * BSIZE + i] /= shadow[i * BSIZE + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (tx>i)
		{
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
		array_offset = offset*matrix_dim+offset;
		#pragma unroll 1 //to disable auto unroll which results in explosion of ports to external memory
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
			m[array_offset+idx] =  peri_col[i*BSIZE+idx];
			array_offset += matrix_dim;
		}
	}
}

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
	for (i=0; i < BSIZE; i++)
	{
		sum += peri_col[ty * BSIZE + i] * peri_row[i * BSIZE + tx];
	}
	m[(global_row_id+ty)*matrix_dim+global_col_id+tx] -= sum;
}
