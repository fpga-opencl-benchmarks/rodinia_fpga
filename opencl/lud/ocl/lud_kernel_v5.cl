#include "../common/opencl_kernel_common.h"

#define REDUCE_LATENCY 8
#define BLOCK_SIZE 8
#define BLOCK_SIZE_LOG 3

#define GA(i,j,block_offset) block_offset + i*size + j						// calculates global address in the matrix
#define LA(i,j)              (i << BLOCK_SIZE_LOG) + j						// calculates local address in the block

__kernel void lud(__global float* RESTRICT a, int size)
{
	int i, j, k, m;
	int offset, block_offset, block_offset_row, block_offset_col, block_offset_chunk;
	int chunk_id, chunk_id_x, chunk_id_y, chunk_size;
	float shift_reg1[REDUCE_LATENCY+1], shift_reg2[REDUCE_LATENCY+1];
	float sum, temp;
	float mem_dia[BLOCK_SIZE*BLOCK_SIZE], mem_row[BLOCK_SIZE*BLOCK_SIZE], mem_col[BLOCK_SIZE*BLOCK_SIZE];

	for (offset = 0; offset < size; offset = offset + BLOCK_SIZE)
	{
		block_offset = offset*size + offset;						// the block is always on the main diagonal

		//======================================================================
		// diagonal block
		//======================================================================

		// loading block into on-chip memory
		for (i = 0; i < BLOCK_SIZE; i++)
		{
			#pragma unroll
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				mem_dia[LA(i,j)] = a[GA(i,j,block_offset)];
			}
		}

		// computation on block
		for (i = 0; i < BLOCK_SIZE; i++)
		{
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				// initiliaze shift register
				#pragma unroll
				for (m = 0; m < REDUCE_LATENCY+1; m++)
				{
					shift_reg1[m] = 0;
				}

				for (k = 0; k < BLOCK_SIZE; k++)
				{
					// reduction
					if (j >= i && k < i)
					{
						shift_reg1[REDUCE_LATENCY] = shift_reg1[0] - mem_dia[LA(i,k)]*mem_dia[LA(k,j)]*1.0;
					}
					else
					{
						shift_reg1[REDUCE_LATENCY] = shift_reg1[0] - mem_dia[LA(i,k)]*mem_dia[LA(k,j)]*0;
					}

					// shifting
					#pragma unroll
					for (m = 0; m < REDUCE_LATENCY; m++)
					{
						shift_reg1[m] = shift_reg1[m+1];
					}
				}

				// final reduction
				#pragma unroll
				for (m = 0; m < REDUCE_LATENCY; m++)
				{
					mem_dia[LA(i,j)] = mem_dia[LA(i,j)] + shift_reg1[m];
				}
			}

			// in the next loop, the value of mem_dia[LA(i,i)] is fixed (for each i) and is reused j times and never overwritten
			temp = mem_dia[LA(i,i)];
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				// initiliaze shift register
				#pragma unroll
				for (m = 0; m < REDUCE_LATENCY+1; m++)
				{
					shift_reg1[m] = 0;
				}

				for (k = 0; k < BLOCK_SIZE; k++)
				{
					// reduction
					if (j >= i+1 && k < i)
					{
						shift_reg1[REDUCE_LATENCY] = shift_reg1[0] - mem_dia[LA(j,k)]*mem_dia[LA(k,i)]*1.0;
					}
					else
					{
						shift_reg1[REDUCE_LATENCY] = shift_reg1[0] - mem_dia[LA(j,k)]*mem_dia[LA(k,i)]*0;
					}

					// shifting
					#pragma unroll
					for (m = 0; m < REDUCE_LATENCY; m++)
					{
						shift_reg1[m] = shift_reg1[m+1];
					}
				}

				// final reduction
				sum = 0;
				#pragma unroll
				for (m = 0; m < REDUCE_LATENCY; m++)
				{
					sum = sum + shift_reg1[m];
				}

				if (j >= i+1)
				{
					mem_dia[LA(j,i)] = (mem_dia[LA(j,i)] + sum)/temp;
				}
				else
				{
					mem_dia[LA(j,i)] = (mem_dia[LA(j,i)] + sum)/1.0;
				}
			}
		}

		// writing back to global memory
		for (i = 1; i < BLOCK_SIZE; i++)						// first row is unchanged, we will not write it back
		{
			#pragma unroll
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				a[GA(i,j,block_offset)] = mem_dia[LA(i,j)];
			}
		}

		//======================================================================
		// perimeter blocks
		//======================================================================

		// the content of the diag local buffer will be reused here
		// we want to process two perimeter blocks in parallel, one in the current row and one in the current column
		// reason why we use two separate local buffers

		chunk_size = ((size - offset) >> BLOCK_SIZE_LOG) - 1;				// offset starts from zero, hence the "-1"
		for (chunk_id = 0; chunk_id < chunk_size; chunk_id++)
		{
			block_offset_row = block_offset + ((chunk_id+1) << BLOCK_SIZE_LOG);	// chunk_id starts from zero, hence the "+1"
			block_offset_col = block_offset + ((chunk_id+1) << BLOCK_SIZE_LOG) * size;

			// loading two block into on-chip memory
			// two separate loops on j are used here to ensure correct coalescing
			for (i = 0; i < BLOCK_SIZE; i++)
			{
				#pragma unroll
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					mem_row[LA(i,j)] = a[GA(i,j,block_offset_row)];		// the "chunk_id+1"th block to the right of the current diagonal block
				}
				#pragma unroll
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					mem_col[LA(i,j)] = a[GA(i,j,block_offset_col)];		// the "chunk_id+1"th block to the bottom of the current diagonal block
				}
			}

			// in the following two loops, since nothing happens on mem_row when i == 0 and only the division part
			// ...is applied on mem_col, we take the computation of i == 0 out of the second loop and put it here
			// ...to avoid all the unneeded shifting and everything on the shift registers, and start the second loop from i = 1
			temp = mem_dia[LA(0,0)];						// i = 0
			for (j = 0; j < BLOCK_SIZE; j++)
			{
				mem_col[LA(j,0)] = mem_col[LA(j,0)]/temp;
			}

			for (i = 1; i < BLOCK_SIZE; i++)
			{
				// same as before, the value of mem_dia[LA(i,i)] is fixed (for each i) and is reused j times and never overwritten
				temp = mem_dia[LA(i,i)];
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					// initiliaze shift register
					#pragma unroll
					for (m = 0; m < REDUCE_LATENCY+1; m++)
					{
						shift_reg1[m] = 0;
						shift_reg2[m] = 0;
					}

					for (k = 0; k < BLOCK_SIZE; k++)
					{
						// reduction
						if (k < i)
						{
							shift_reg1[REDUCE_LATENCY] = shift_reg1[0] - mem_dia[LA(i,k)]*mem_row[LA(k,j)]*1.0;
							shift_reg2[REDUCE_LATENCY] = shift_reg2[0] - mem_dia[LA(k,i)]*mem_col[LA(j,k)]*1.0;
						}
						else
						{
							shift_reg1[REDUCE_LATENCY] = shift_reg1[0] - mem_dia[LA(i,k)]*mem_row[LA(k,j)]*0;
							shift_reg2[REDUCE_LATENCY] = shift_reg2[0] - mem_dia[LA(k,i)]*mem_col[LA(j,k)]*0;
						}

						// shifting
						#pragma unroll
						for (m = 0; m < REDUCE_LATENCY; m++)
						{
							shift_reg1[m] = shift_reg1[m+1];
							shift_reg2[m] = shift_reg2[m+1];
						}
					}

					// final reduction
					sum = 0;
					#pragma unroll
					for (m = 0; m < REDUCE_LATENCY; m++)
					{
						mem_row[LA(i,j)] = mem_row[LA(i,j)] + shift_reg1[m];	// no need to put this in a temp sum variable
						sum = sum + shift_reg2[m];				// this one needs sum variable due to division outside of the loop
					}
					mem_col[LA(j,i)] = (mem_col[LA(j,i)] + sum)/temp;
				}
			}

			// writing back to global memory
			for (i = 1; i < BLOCK_SIZE; i++)						// first row has not changed, hence we start from i = 1
			{
				#pragma unroll
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					a[GA(i,j,block_offset_row)] = mem_row[LA(i,j)];
				}
			}
			for (i = 0; i < BLOCK_SIZE; i++)
			{
				#pragma unroll
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					a[GA(i,j,block_offset_col)] = mem_col[LA(i,j)];
				}
			}
		}

		//======================================================================
		// internal blocks
		//======================================================================

		// here to calculate each block, we need the topmost block in the same column and the leftmost block in the row
		// movement on blocks is row by row which allows us to reuse the content of the mem_col buffer for each row
		// mem_col and mem_row here do NOT refer to the necessary block on the same column and row, but rather, the exact opposite of it
		// ...this is the result of the naming sceme we used in calculating the perimeter blocks

		for (chunk_id_y = 0; chunk_id_y < chunk_size; chunk_id_y++)
		{
			block_offset_col = block_offset + ((chunk_id_y+1) << BLOCK_SIZE_LOG) * size;							// for leftmost block in the same block row

			// loading leftmost block in the same block row into on-chip memory
			for (i = 0; i < BLOCK_SIZE; i++)
			{
				#pragma unroll
				for (j = 0; j < BLOCK_SIZE; j++)
				{
					mem_col[LA(i,j)] = a[GA(i,j,block_offset_col)];
				}
			}

			for (chunk_id_x = 0; chunk_id_x < chunk_size; chunk_id_x++)
			{
				block_offset_row = block_offset + ((chunk_id_x+1) << BLOCK_SIZE_LOG);							// for topmost block in the same block column
				block_offset_chunk = block_offset + ((chunk_id_y+1) << BLOCK_SIZE_LOG) * size + ((chunk_id_x+1) << BLOCK_SIZE_LOG);	// for current block

				// loading topmost block in the same block column and current block into on-chip memory
				for (i = 0; i < BLOCK_SIZE; i++)
				{
					#pragma unroll
					for (j = 0; j < BLOCK_SIZE; j++)
					{
						mem_row[LA(i,j)] = a[GA(i,j,block_offset_row)];
					}
					#pragma unroll
					for (j = 0; j < BLOCK_SIZE; j++)
					{
						mem_dia[LA(i,j)] = a[GA(i,j,block_offset_chunk)];
					}
				}

				for (i = 0; i < BLOCK_SIZE; i++)
				{
					#pragma unroll
					for (j = 0; j < BLOCK_SIZE; j++)
					{
						// since the following loop is fully unrolled, it doesn't need reduction
						#pragma unroll
						for (k = 0; k < BLOCK_SIZE; k++)
						{
							mem_dia[LA(i,j)] = mem_dia[LA(i,j)] - mem_col[LA(i,k)]*mem_row[LA(k,j)];
						}
					}
				}

				// wrting current chunk back to global memory
				for (i = 0; i < BLOCK_SIZE; i++)
				{
					#pragma unroll
					for (j = 0; j < BLOCK_SIZE; j++)
					{
						a[GA(i,j,block_offset_chunk)] = mem_dia[LA(i,j)];
					}
				}
			}
		}
	}
}
