#ifdef USE_AOT
	#include "problem_size.h"
#endif

#define LMEM_SIZE __attribute__((local_mem_size(BLOCK_SIZE*BLOCK_SIZE*4))) 

#include "../common/opencl_kernel_common.h"

__kernel void lud_diagonal(__global float* RESTRICT m, 
                 LMEM_SIZE __local  float* RESTRICT shadow,
                                    int             matrix_dim, 
                                    int             offset)
{ 
	int i, j, k;
	int sub_matrix_offset;

	sub_matrix_offset = offset * matrix_dim + offset;
	for(i = 0; i < BLOCK_SIZE; i++)
	{
		for (j = 0; j < BLOCK_SIZE; j++)
		{
			shadow[i * BLOCK_SIZE + j] = m[sub_matrix_offset + j];
		}
		sub_matrix_offset += matrix_dim;
	}
  
	for(i = 0; i < BLOCK_SIZE-1; i++)
	{
		for (j = i+1; j < BLOCK_SIZE; j++)
		{
			for(k = 0; k < i; k++)
			{
				shadow[j * BLOCK_SIZE + i] -= shadow[j * BLOCK_SIZE + k] * shadow[k * BLOCK_SIZE + i];
			}
			shadow[j * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
		}
		
		for (j = i+1; j < BLOCK_SIZE; j++)
		{
			for(k = 0; k < i+1; k++)
			{
				shadow[(i+1) * BLOCK_SIZE + j] -= shadow[(i+1) * BLOCK_SIZE + k] * shadow[k * BLOCK_SIZE + j];
			}
		}
	}

	sub_matrix_offset = (offset+1) * matrix_dim + offset;
	for(i = 1; i < BLOCK_SIZE; i++)
	{
		for (j = 0; j < BLOCK_SIZE; j++)
		{
			m[sub_matrix_offset+j] = shadow[i * BLOCK_SIZE + j];
		}
		sub_matrix_offset += matrix_dim;
	}
}

__kernel void lud_perimeter(__global float* RESTRICT m, 
                  LMEM_SIZE __local  float* RESTRICT dia,
                  LMEM_SIZE __local  float* RESTRICT peri_row,
                  LMEM_SIZE __local  float* RESTRICT peri_col,
                                     int             matrix_dim, 
                                     int             offset)
{
	int i, j, k;
	int sub_matrix_offset, block_offset;
	int chunk_id, chunk_num;

	chunk_num = ((matrix_dim - offset) / BLOCK_SIZE) - 1;
	sub_matrix_offset = offset * matrix_dim + offset;

	for (chunk_id = 0; chunk_id < chunk_num; chunk_id++)
	{
		// load data from global to local memory
		block_offset = sub_matrix_offset;
		for (i = 0; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				dia[i * BLOCK_SIZE + j] = m[block_offset + j];
			}
			block_offset += matrix_dim;
		}

		block_offset = sub_matrix_offset + (chunk_id+1) * BLOCK_SIZE;
		for (i = 0; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				peri_row[i * BLOCK_SIZE + j] = m[block_offset + j];
			}
			block_offset += matrix_dim;
		}

		block_offset = sub_matrix_offset + (chunk_id+1) * BLOCK_SIZE * matrix_dim;
		for (i = 0; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				peri_col[i * BLOCK_SIZE + j] = m[block_offset + j];
			}
			block_offset += matrix_dim;
		}

		// compute
		// peri-row
		for(i = 1; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				for (k = 0; k < i; k++)
				{
					peri_row[i * BLOCK_SIZE + j] -= dia[i * BLOCK_SIZE + k] * peri_row[k * BLOCK_SIZE + j];
				}
			}
		}

		// peri-col
		for(i = 0; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				for(k = 0; k < i; k++)
				{
					peri_col[j * BLOCK_SIZE + i] -= dia[k * BLOCK_SIZE + i] * peri_col[j * BLOCK_SIZE + k];
				}
				peri_col[j * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];
			}
		}

		// write back to global memory
		// peri-row
		block_offset = sub_matrix_offset + (chunk_id+1) * BLOCK_SIZE + matrix_dim;
		for(i = 1; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				m[block_offset + j] = peri_row[i * BLOCK_SIZE + j];
			}
			block_offset += matrix_dim;
		}

		// peri-col
		block_offset = sub_matrix_offset + (chunk_id+1) * BLOCK_SIZE * matrix_dim;
		for(i = 0; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				m[block_offset + j] = peri_col[i * BLOCK_SIZE + j];
			}
			block_offset += matrix_dim;
		}
	}
}

__kernel void lud_internal(__global float* RESTRICT m, 
                 LMEM_SIZE __local  float* RESTRICT peri_row,
                 LMEM_SIZE __local  float* RESTRICT peri_col,
                                    int             matrix_dim, 
                                    int             offset)
{
	int i, j, k, chunk_id_x, chunk_id_y, chunk_num;
	int sub_matrix_offset, block_offset, block_offset_row, block_offset_col;
	float sum;

	chunk_num = ((matrix_dim - offset) / BLOCK_SIZE) - 1;
	sub_matrix_offset = offset * matrix_dim + offset;

	for (chunk_id_y = 0; chunk_id_y < chunk_num; chunk_id_y++)
	{
		block_offset_col = sub_matrix_offset + (chunk_id_y + 1) * BLOCK_SIZE * matrix_dim;

		for (i = 0; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				peri_col[i * BLOCK_SIZE + j] = m[block_offset_col + j];
			}
			block_offset_col += matrix_dim;
		}

		for (chunk_id_x = 0; chunk_id_x < chunk_num; chunk_id_x++)
		{
			block_offset_row = sub_matrix_offset + (chunk_id_x + 1) * BLOCK_SIZE;
			block_offset = sub_matrix_offset + (chunk_id_y + 1) * BLOCK_SIZE * matrix_dim + (chunk_id_x + 1) * BLOCK_SIZE;

			for (i = 0; i < BLOCK_SIZE; i++)
			{
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					peri_row[i * BLOCK_SIZE + j] = m[block_offset_row + j];
				}
				block_offset_row += matrix_dim;
			}

			for (i = 0; i < BLOCK_SIZE; i++)
			{
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					sum = 0;
					for (k = 0; k < BLOCK_SIZE; k++)
					{
						sum += peri_col[i * BLOCK_SIZE + k] * peri_row[k * BLOCK_SIZE + j];
					}
					m[block_offset + j] -= sum;
				}
				block_offset += matrix_dim;
			}
		}
	}
}
