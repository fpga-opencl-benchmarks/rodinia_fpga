#include "hotspot_common.h"
#include "../common/opencl_kernel_common.h"

#ifndef BSIZE
	#ifdef AOCL_BOARD_de5net_a7
		#define BSIZE 512
	#elif  AOCL_BOARD_a10pl4_dd4gb_gx115es3
		#define BSIZE 1024
	#else
		#define BSIZE 512
	#endif
#endif

#ifndef SSIZE
	#ifdef AOCL_BOARD_de5net_a7
		#define SSIZE 8
	#elif  AOCL_BOARD_a10pl4_dd4gb_gx115es3
		#define SSIZE 8
	#else
		#define SSIZE 4
	#endif
#endif

#pragma OPENCL EXTENSION cl_altera_channels : enable;
// the BSIZE/SSIZE channel depth ensures that the odd kernel will never be more than one block behind the even kernel and hence,
// storing boundary data for two blocks in the __global boundary buffer is enough for correct operation without values in that
// buffer being overwritten by the even kernel, before being read by the odd kernel
channel float data_ch[SSIZE] __attribute__((depth(BSIZE/SSIZE)));
channel float power_ch[SSIZE] __attribute__((depth(BSIZE/SSIZE)));

#define SW_COLS            BSIZE
#define SW_BASE_SIZE_ODD   SW_COLS * 2
#define SW_BASE_SIZE_EVEN  (SW_COLS * 2) + 1					// base point in the even kernel is one point further since x starts from -1 instead of 0
#define SW_SIZE_ODD        SW_BASE_SIZE_ODD + SSIZE
#define SW_SIZE_EVEN       SW_BASE_SIZE_EVEN + SSIZE				// the window in the even kernel is one point bigger since x starts from -1 instead of 0

__attribute__((max_global_work_dim(0)))
__kernel void hotspot_even(__global const float* RESTRICT power,		// power input
                           __global const float* RESTRICT src,			// temperature input
                                          int             grid_cols,		// number of columns
                                          int             grid_rows,		// number of rows
                                          float           step_div_Cap,		// step/Capacitance
                                          float           Rx_1,
                                          float           Ry_1,
                                          float           Rz_1,
                           __global       float* RESTRICT boundary,
                                          int             rem_iter)		// number of remaining iterations)
{
	float sw[SW_SIZE_ODD];

	// initialize
	#pragma unroll
	for (int i = 0; i < SW_SIZE_ODD; i++)
	{
		sw[i] = 0.0f;
	}

	// read index
	int x = 0;
	int y = -1;	// y starts from -1 so that for first row, both rows of the sliding window are filled by the same data, one for top neighbor and the other for current index
	int bx = 0;

	do
	{
		int gx = bx + x;								// global x position
		int comp_offset_y = y - 1;							// global y position, will be out-of-bound for first two and last iterations
		int comp_offset = gx + comp_offset_y * grid_cols;				// global index, will be out-of-bound for first two and last iterations
		int read_offset_y = (y < 0) ? 0 : ((y == grid_rows) ? grid_rows - 1 : y);	// out-of-bound reads will be reset to boundary (y == -1, y == 0 & y == grid_rows)
		int read_offset = gx + read_offset_y * grid_cols;				// real index to read from memory, will never be out-of-bound
      
		// shift
		#pragma unroll
		for (int i = 0; i < SW_BASE_SIZE_ODD; i++)
		{
			sw[i] = sw[i + SSIZE];
		}

		int sw_offset   = SW_COLS;
		int sw_offset_s = sw_offset + SW_COLS;
		int sw_offset_n = sw_offset - SW_COLS;
		int sw_offset_e = sw_offset + 1;
		int sw_offset_w = sw_offset - 1;

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			sw[SW_BASE_SIZE_ODD + i] = src[read_offset + i];			// read new values from memory

			if (gx + i < grid_cols && comp_offset_y != -1 && comp_offset_y != -2 && comp_offset_y != grid_rows) // to avoid going out of bounds in either axes
			{
				float north, south, east, west, current, local_power;

				// left neighbor
				if (i == 0 && gx == 0)						// leftmost column in matrix
				{
					west = sw[sw_offset];
				}
				else if (i == 0 && x == 0)					// left block boundary
				{
					west = src[comp_offset - 1];
				}
				else
				{
					west = sw[sw_offset_w + i];
				}

				// right neighbor
				if (gx + i == grid_cols - 1)					// rightmost column in matrix
				{
					east = sw[sw_offset + i];
				}
				else if (i == SSIZE - 1 && x + SSIZE == BSIZE)			// right block boundaries, the clause is also equal to x + i == BSIZE - 1 but for some reason, using that results in a lot more global memory read ports and much higher area usage!! 
				{
					east = src[comp_offset + SSIZE];
				}
				else
				{
					east = sw[sw_offset_e + i];
				}

				north       = sw[sw_offset_n + i];				// top neighbor
				south       = sw[sw_offset_s + i];				// bottom neighbor
				current     = sw[sw_offset   + i];				// current index
				local_power = power[comp_offset + i];

				float v = local_power                             +
					  (east  + west  - 2.0f * current) * Rx_1 +
					  (north + south - 2.0f * current) * Ry_1 +
					  (AMB_TEMP - current)             * Rz_1;

				float delta = step_div_Cap * v;
				float output = current + delta;
				if ((x == BSIZE - SSIZE) && (i == 6 || i == 7))			// last two columns of the block
				{
					// the below offset ensures that boundary values from first block are written to addresses 0 -> (2 * grid_rows) - 1, values from second block
					// are written to 2 * grid_rows --> (4 * grid_rows) - 1, and after that, even-numbered and odd-numbered blocks will overwrite values from the
					// first and second blocks, respectively
					int boundary_offset = ((bx/BSIZE) % 2) * 2 * grid_rows;
					boundary[boundary_offset + 2 * comp_offset_y + (i - 6)] = output; // write a copy of these to global memory to be used on the first and second columns of next block in the "odd" kernel
				}

				write_channel_altera(data_ch[i], output);			// write output to channel instead of global memory to be used in the "odd" kernel

				if (!(i == SSIZE - 1 && x + SSIZE == BSIZE) && rem_iter != 1)	// if not at the last column of block or last iteration if number of iterations is odd
				{
					// write power value to channel to be read in the odd kernel, this will significantly decrease memory access in the odd kernel
					// since the odd kernel is always one column behind the even kernel, iteration i should write to channel i + 1 (and 7 should write to zero)
					write_channel_altera(power_ch[(i + 1) % SSIZE], local_power);
				}
			}
		}

		x = (x + SSIZE) % BSIZE;							// resets x to zero when x == BSIZE
		y = (x == 0) ? ((y == grid_rows) ? -1 : y + 1) : y;				// x = 0 is the end of block since value of x has already been updated to the next value

		if (x == 0 && y == -1)
		{
			bx += BSIZE;								// go to next block
		}
	} while (bx + x < grid_cols);
}

__attribute__((max_global_work_dim(0)))
__kernel void hotspot_odd(__global const float* RESTRICT power,			// power input
                          __global       float* RESTRICT dst,			// temperature output
                                         int             grid_cols,		// number of columns
                                         int             grid_rows,		// number of rows
                                         float           step_div_Cap,		// step/Capacitance
                                         float           Rx_1,
                                         float           Ry_1,
                                         float           Rz_1, 
                          __global       float* RESTRICT boundary,
                                         int             rem_iter)		// number of remaining iterations
{
	float sw[SW_SIZE_EVEN];

	// initialize
	#pragma unroll
	for (int i = 0; i < SW_SIZE_EVEN; i++)
	{
		sw[i] = 0.0f;
	}

	// read index
	int x = -1;	// unlike the "even" kernel, x starts from -1 since this kernel cannot process the rightmost column of each block due to its right neighbor having not been calculated yet and hence, this kernel is always one column behind the "even" kernel
	int y = 0;	// unlike the "even" kernel, y starts from 0 since we are reading the input from a channel and not global memory and cannot read the first row twice
	int bx = 0;

	do
	{
		int gx = bx + x;								// global x position
		int comp_offset_y = y - 1;							// global y position, will be out-of-bound for first two and last iterations
		int comp_offset = gx + comp_offset_y * grid_cols;				// global index, will be out-of-bound for first and last iterations
		float tmp_boundary[2];								// local copy of boundary values

		// here, the first block does not need to read anything from the boundary buffer
		// the below offset ensures that boundary values for odd-numbered blocks are read from addresses 0 -> (2 * grid_rows) - 1 and values for
		// even-numbered blocks are read from 2 * grid_rows --> (4 * grid_rows) - 1
		int boundary_offset = (((bx/BSIZE) + 1) % 2) * 2 * grid_rows;
      
		// shift
		#pragma unroll
		for (int i = 0; i < SW_BASE_SIZE_EVEN; i++)
		{
			sw[i] = sw[i + SSIZE];
		}

		if (bx != 0)									// anywhere other than first block
		{
			#pragma unroll
			for (int i = 0; i < 2; i++)
			{
				tmp_boundary[i] = boundary[boundary_offset + 2 * comp_offset_y + i]; // coalesced read of the necessary backed up values for boundary columns
			}
		}

		int sw_offset   = SW_COLS;
		int sw_offset_s = sw_offset + SW_COLS;
		int sw_offset_n = sw_offset - SW_COLS;
		int sw_offset_e = sw_offset + 1;
		int sw_offset_w = sw_offset - 1;

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			if (gx + i < grid_cols - 1 && comp_offset_y != grid_rows - 1)		// there is nothing new to read on rightmost row and bottommost column
			{
				sw[SW_BASE_SIZE_EVEN + i] = read_channel_altera(data_ch[i]);	// read new values from channel
			}

			if (!(gx == -1 && i == 0) && gx + i < grid_cols && comp_offset_y != -1 && comp_offset_y != grid_rows) // avoid going out of bounds in either axes
			{
				float north, south, east, west, current, local_power, delta = 0;

				// current index
				if (i == 0 && x == -1)						// first column in block
				{
					current = tmp_boundary[1];				// read from backup
				}
				else
				{
					current = sw[sw_offset + i];
				}

				if (rem_iter != 1)						// no need to compute delta if number of iterations is odd and we are at last iteration
				{
					// left neighbor
					if (i == 1 && gx == -1)					// leftmost column in matrix
					{
						west = current;					// read current index on matrix boundary
					}
					else if (i == 0 && x == -1)				// first column in block
					{
						west = tmp_boundary[0];				// read from backup
					}
					else if (i == 1 && x == -1)				// second column in block
					{
						west = tmp_boundary[1];				// read from backup
					}
					else
					{
						west = sw[sw_offset_w + i];
					}

					// right neighbor
					// in the following, the right neighbor is always read from the sliding window since the block in this kernel is one column behind the block in the "even" kernel
					if (gx + i == grid_cols - 1)				// rightmost column in matrix
					{
						east = current;					// read current index on matrix boundary
					}
					else
					{
						east = sw[sw_offset_e + i];
					}

					// top neighbor
					if (comp_offset_y == 0)					// top matrix boundary
					{
						north = current;				// read current index on matrix boundary
					}
					else if (i == 0 && x == -1)				// first column in block
					{
						north = boundary[boundary_offset + 2 * (comp_offset_y - 1) + 1]; // read from backup
					}
					else
					{
						north = sw[sw_offset_n + i];
					}

					// bottom neighbor
					if (comp_offset_y == grid_rows - 1)			// bottom matrix boundary
					{
						south = current;				// read current index on matrix boundary
					}
					else if (i == 0 && x == -1)				// first column in block
					{
						south = boundary[boundary_offset + 2 * (comp_offset_y + 1) + 1]; // read from backup
					}
					else
					{
						south = sw[sw_offset_s + i];
					}

					if (i == 0 && x == -1)					// first column in block
					{
						local_power = power[comp_offset + i]; 		// read directly from off-chip memory
					}
					else
					{
						local_power = read_channel_altera(power_ch[i]);	// read from channel
					}

					float v = local_power               		  +
						  (east  + west  - 2.0f * current) * Rx_1 +
						  (north + south - 2.0f * current) * Ry_1 +
						  (AMB_TEMP - current)             * Rz_1;

					delta = step_div_Cap * v;
				}

				dst[comp_offset + i] = current + delta;				// if number of iterations is odd and we are at last iteration, delta will be zero and we just write the input from the even kernel to memory
			}
		}

		// the commented lines below show how this part of the code was originally written
		// after lots of experimentation, the (gx + SSIZE > grid_cols) condition by itself turned out
		// to be the reason why the operating frequency of the kernel was so low, and removing that condition
		// without touching the rest of the code resulted in the operating frequency going up from ~150 MHz to ~250 Mhz
		// the reason for this is not clear, but obviously that condition is necessary for correct operation
		// because of this, after trying 10-15 different ways to code the following lines, the below implementation
		// proved to give the highest operating frequency without sacrificing much area
		// still, the final operating frequency of the implementation is less than when the condition is completely removed

		/*x = ((x + SSIZE == BSIZE - 1) || (gx + SSIZE > grid_cols)) ? -1 : x + SSIZE;	// x is reset to -1 in two cases; one is when a block is completely traversed, and the other is when the index goes out-of-bound when grid size is not a multiple of block size
		y = (x == -1) ? ((y == grid_rows) ? 0 : y + 1) : y;				// x = -1 is the end of block since value of x has already been updated to the next value

		if (x == -1 && y == 0)
		{
			bx += BSIZE;								// go to next block
		}*/

		x = x + SSIZE;
		if ((x == BSIZE - 1) || (gx == grid_cols - (grid_cols % SSIZE) - 1))		// x is reset to -1 in two cases; one is when a block is completely traversed, and the other is when the index goes out-of-bound if grid size is not a multiple of block size
		{
			x = -1;
			if (y == grid_rows)							// if on last row
			{
				y = 0;								// reset row number
				bx += BSIZE;							// got to next block
			}
			else
			{
				y = y + 1;							// go to next row
			}
		}
	} while (bx + x < grid_cols);
}
