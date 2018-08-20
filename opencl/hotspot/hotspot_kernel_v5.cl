#include "hotspot_common.h"

// input shift register parameters
#define IN_SR_BASE		2 * BSIZE				// this shows the point to write into the shift register; one row for top neighbors and one row for center
#define IN_SR_SIZE		IN_SR_BASE + SSIZE		// SSIZE indexes are enough for the bottommost row

// offsets for reading from the input shift register
#define SR_OFF_C		BSIZE				// center
#define SR_OFF_N		SR_OFF_C + BSIZE		// north  (+y direction)
#define SR_OFF_S		SR_OFF_C - BSIZE		// south  (-y direction)
#define SR_OFF_E		SR_OFF_C + 1			// east   (-x direction)
#define SR_OFF_W		SR_OFF_C - 1			// west   (+x direction)

__attribute__((max_global_work_dim(0)))
__kernel void hotspot(__global const float* restrict pwr_in,		// power input
                      __global const float* restrict src,			// temperature input
                      __global       float* restrict dst,			// temperature output
                               const int             grid_cols,		// number of columns
                               const int             grid_rows,		// number of rows
                               const float           sdc,			// step/Capacitance
                               const float           Rx_1,
                               const float           Ry_1,
                               const float           Rz_1,
                               const int             comp_exit)		// exit condition for compute loop
{
	float in_sr[IN_SR_SIZE];									// shift register for spatial blocking

	// initialize the shift register
	#pragma unroll
	for (int i = 0; i < IN_SR_SIZE; i++)
	{
		in_sr[i] = 0.0f;
	}

	// starting points
	int x = 0;
	int y = 0;
	int bx = 0;											// block number in x dimension
	int index = 0;											// global index

	while (index != comp_exit)
	{
		index++;											// increment global index
		
		// shift shift register by SSIZE
		#pragma unroll
		for (int i = 0; i < IN_SR_BASE; i++)
		{
			in_sr[i] = in_sr[i + SSIZE];
		}

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - 1;							// global x position offset
			int real_x = gx + i;							// global x position
			int comp_offset_y = y - 1;						// global y position, will be out-of-bound for first and last iterations
			int comp_offset = gx + comp_offset_y * grid_cols;		// index offset for computation, reading from power buffer, and writing back to destination
			int read_offset = comp_offset + grid_cols;			// index to read from memory
			int real_block_x = x + i - 1;						// local x position in block, adjusted for halo

			float north, south, east, west, center, local_pwr;

			// read input values from memory
			// avoid out-of-bound indexes in x direction
			if (real_x >= 0 && real_x < grid_cols)				// avoid out-of-bound indexes in x direction
			{
				// nothing to read on last row
				if (comp_offset_y != grid_rows - 1)
				{
					in_sr[IN_SR_BASE + i]  = src[read_offset + i];// read new temperature values from memory
				}

				// there is nothing to compute on the first out-of-bound row
				if (comp_offset_y >= 0)
				{
					local_pwr = pwr_in[comp_offset + i];	// read new power values from memory
				}
			}
			
			center = in_sr[SR_OFF_C + i];						// center index

			// west neighbor
			if (real_x == 0)								// leftmost column in matrix
			{
				west = center;								// fall back on center
			}
			else
			{
				west = in_sr[SR_OFF_W + i];					// read from shift register
			}

			// east neighbor
			if (real_x == grid_cols - 1)						// rightmost column in matrix
			{
				east = center;								// fall back on center
			}
			else
			{
				east = in_sr[SR_OFF_E + i];					// read from shift register
			}

			// top neighbor
			if (comp_offset_y == grid_rows - 1)				// topmost row, deliberately written like this to align with v1 kernel
			{
				north = center;							// fall back on center
			}
			else
			{
				north = in_sr[SR_OFF_N + i];					// read from shift register
			}

			// bottom neighbor
			if (comp_offset_y == 0)							// bottommost row, deliberately written like this to align with v1 kernel
			{
				south = center;							// fall back on center
			}
			else
			{
				south = in_sr[SR_OFF_S + i];					// read from shift register
			}

			float v = local_pwr                              +
					(north + south - 2.0f * center) * Ry_1 +
					(east  + west  - 2.0f * center) * Rx_1 +
					(AMB_TEMP - center)             * Rz_1;

			float delta = sdc * v;

			// write output values to memory
			// the following condition is to avoid halos and going out of bounds in all axes
			if (comp_offset_y >= 0 && real_block_x >= 0 && real_block_x < BSIZE - 2 && real_x < grid_cols)
			{
				dst[comp_offset + i] = center + delta;
			}

			//if (ID == 1 && comp_offset_y >=0 && comp_offset_y < grid_rows && real_x >=0 && real_x < grid_cols)
				//printf("ID: %d, row: %04d, col: %04d, center: %f, top: %f, bottom: %f, left: %f, right: %f, pow: %f, out: %f\n", ID, comp_offset_y, real_x, center, north, south, west, east, local_pwr, out.data[i]);
		}

		// equivalent to x = (x + SSIZE) % BSIZE
		x = (x + SSIZE) & (BSIZE - 1);						// move one chunk forward and reset to zero if end of row was reached

		if (x == 0)										// if one row finished
		{
			if (y == grid_rows)								// if on last row (compute traverses one more plane than memory read/write)
			{
				y = 0;									// reset row number
				bx += BSIZE - 2;							// go to next block, account for overlapping
			}
			else
			{
				y++;										// go to next row
			}
		}
	}
}
