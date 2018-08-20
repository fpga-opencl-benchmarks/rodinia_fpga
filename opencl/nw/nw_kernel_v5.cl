#include "work_group_size.h"

__attribute__((max_global_work_dim(0)))
__kernel void nw_kernel1(__global int* restrict reference, 
                         __global int* restrict data,
                         __global int* restrict input_v,				// vertical input (first column)
                                  int           dim,
                                  int           penalty,
                                  int           loop_exit,
                                  int           block_offset)
{
	int out_SR[PAR - 1][3];										// output shift register; 2 registers per parallel comp_col_offset is required, one for writing, and two for passing data to the following diagonal lines to handle the dependency
	int last_chunk_col_SR[BSIZE - (PAR - 1) + 2];					// shift register for last comp_col_offset in parallel chunk to pass data to next chunk, (PAR - 1) cells are always out of bound, one extra cell is added for writing and one more for top-left
	int ref_SR[PAR][PAR];										// shift registers to align reads from the reference buffer
	int data_h_SR[PAR][PAR];										// shift registers to align reading the first comp_row of data buffer
	int write_SR[PAR][PAR];										// shift registers to align writes to external memory
	int input_v_SR[2];											// one for left, one for top-left

	// initialize shift registers
	#pragma unroll
	for (int i = 0; i < PAR - 1; i++)
	{
		#pragma unroll
		for (int j = 0; j < 3; j++)
		{
			out_SR[i][j] = 0;
		}
	}
	#pragma unroll
	for (int i = 0; i < BSIZE - PAR + 3; i++)
	{
		last_chunk_col_SR[i] = 0;
	}
	#pragma unroll
	for (int i = 0; i < PAR; i++)
	{
		#pragma unroll
		for (int j = 0; j < PAR; j++)
		{
			write_SR[i][j] = 0;
		}
	}
	#pragma unroll
	for (int i = 0; i < PAR; i++)
	{
		#pragma unroll
		for (int j = 0; j < PAR; j++)
		{
			ref_SR[i][j] = 0;
		}
	}
	#pragma unroll
	for (int i = 0; i < PAR; i++)
	{
		#pragma unroll
		for (int j = 0; j < PAR; j++)
		{
			data_h_SR[i][j] = 0;
		}
	}
	#pragma unroll
	for (int i = 0; i < 2; i++)
	{
		input_v_SR[i] = 0;
	}

	// starting points
	int comp_col_offset = 0;
	int write_col_offset = -PAR;
	int block_row = 0;
	int loop_index = 0;

	#pragma ivdep array(data)
	while (loop_index != loop_exit)
	{
		loop_index++;

		// shift the shift registers
		#pragma unroll
		for (int i = 0; i < PAR - 1; i++)
		{
			#pragma unroll
			for (int j = 0; j < 2; j++)
			{
				out_SR[i][j] = out_SR[i][j + 1];
			}
		}
		#pragma unroll
		for (int i = 0; i < BSIZE - PAR + 2; i++)
		{
			last_chunk_col_SR[i] = last_chunk_col_SR[i + 1];
		}
		#pragma unroll
		for (int i = 0; i < PAR; i++)
		{
			#pragma unroll
			for (int j = 0; j < PAR - 1; j++)
			{
				write_SR[i][j] = write_SR[i][j + 1];
			}
		}
		#pragma unroll
		for (int i = 0; i < PAR; i++)
		{
			#pragma unroll
			for (int j = 0; j < PAR - 1; j++)
			{
				ref_SR[i][j] = ref_SR[i][j + 1];
			}
		}
		#pragma unroll
		for (int i = 0; i < PAR; i++)
		{
			#pragma unroll
			for (int j = 0; j < PAR - 1; j++)
			{
				data_h_SR[i][j] = data_h_SR[i][j + 1];
			}
		}
		#pragma unroll
		for (int i = 0; i < 1; i++)
		{
			input_v_SR[i] = input_v_SR[i + 1];
		}

		int read_block_row = block_row;
		int read_row = block_offset + read_block_row;

		if (comp_col_offset == 0 && read_row < dim - 1)
		{
			input_v_SR[1] = input_v[read_row];
		}

		if (block_row == 0)
		{		
			#pragma unroll
			for (int i = 0; i < PAR; i++)
			{
				int read_col = comp_col_offset + i;
				int read_index = read_row * dim + read_col;

				if (read_col < dim - 1 && read_row < dim - 1)
				{
					data_h_SR[i][i] = data[read_index];
				}
			}
		}

		#pragma unroll
		for (int i = PAR - 1; i >= 0; i--)
		{
			int comp_block_row = (BSIZE + block_row - i) & (BSIZE - 1); // read_col > 0 is skipped since it has area overhead and removing it is harmless
			int comp_row = block_offset + comp_block_row;
			int comp_col = comp_col_offset + i;

			int read_col = comp_col_offset + i;
			int read_index = read_row * dim + read_col;

			if (read_row > 0 && read_col < dim - 1 && read_row < dim - 1) // read_col > 0 is skipped since it has area overhead and removing it is harmless
			{
				ref_SR[i][i] = reference[read_index];
			}

			int top      = (i == PAR - 1) ? last_chunk_col_SR[BSIZE - PAR + 1] : out_SR[  i  ][1];
			int top_left = (comp_col_offset == 0 && i == 0) ? input_v_SR[0] : ((i == 0) ? last_chunk_col_SR[0] : out_SR[i - 1][0]);
			int left     = (comp_col_offset == 0 && i == 0) ? input_v_SR[1] : ((i == 0) ? last_chunk_col_SR[1] : out_SR[i - 1][1]);

			int out1 = top_left + ref_SR[i][0];
			int out2 = left - penalty;
			int out3 = top - penalty;
			int max_temp = (out1 > out2) ? out1 : out2;
			int max = (out3 > max_temp) ? out3 : max_temp;

			// directly pass input to output if on the first row in the block which is overlapped with the previous block
			int out = (comp_block_row == 0) ? data_h_SR[i][0] : max;

			if (i == PAR - 1)									// if on last column in chunk
			{
				last_chunk_col_SR[BSIZE - PAR + 2] = out;
			}
			else
			{
				out_SR[i][2] = out;
			}

			write_SR[i][PAR - i - 1] = out;

			int write_col = write_col_offset + i;
			int write_block_row = (BSIZE + block_row - (PAR - 1)) & (BSIZE - 1); // write to memory is always PAR - 1 rows behind compute
			int write_row = block_offset + write_block_row;
			int write_index = write_row * dim + write_col;

			if (write_block_row > 0 && write_col < dim - 1 && write_row < dim - 1)
			{
				data[write_index] = write_SR[i][0];
			}
		}

		block_row = (block_row + 1) & (BSIZE - 1);
		if (block_row == PAR - 1)
		{
			write_col_offset += PAR;
		}
		if (block_row == 0)
		{
			comp_col_offset += PAR;
		}
	}
}
