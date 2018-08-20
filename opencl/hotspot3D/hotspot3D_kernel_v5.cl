#include "hotspot3D_common.h"

// input shift register parameters
#define IN_SR_BASE		2 * BLOCK_X * BLOCK_Y			// this shows the point to write into the shift register; one row for top neighbors and one row for center
#define IN_SR_SIZE		IN_SR_BASE + SSIZE				// SSIZE indexes are enough for the bottommost row

// offsets for reading from the input shift register
#define SR_OFF_C		BLOCK_X * BLOCK_Y				// center
#define SR_OFF_S		SR_OFF_C + BLOCK_X				// south  (-y direction)
#define SR_OFF_N		SR_OFF_C - BLOCK_X				// north  (+y direction)
#define SR_OFF_E		SR_OFF_C + 1					// east   (-x direction)
#define SR_OFF_W		SR_OFF_C - 1					// west   (+x direction)
#define SR_OFF_A		SR_OFF_C + BLOCK_X * BLOCK_Y		// above  (-z direction), this is intentionally written like this to follow the behavior of the baseline implementation
#define SR_OFF_B		SR_OFF_C - BLOCK_X * BLOCK_Y		// below  (+z direction), this is intentionally written like this to follow the behavior of the baseline implementation


__attribute__((max_global_work_dim(0)))
__kernel void hotspotOpt1(__global const float* restrict pwr_in,		// power input
                          __global const float* restrict src,		// temperature input
                          __global       float* restrict dst,		// temperature output
                                   const float           sdc,		// step/Capacitance
                                   const int             nx,		// x dimension size
                                   const int             ny,		// y dimension size
                                   const int             nz,		// z dimension size
                                   const float           ce,
                                   const float           cw, 
                                   const float           cn,
                                   const float           cs,
                                   const float           ct,
                                   const float           cb, 
                                   const float           cc,
                                   const int             last_col,	// exit condition in x direction
                                   const int             comp_exit)	// exit condition for compute loop
{
	// starting points
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;											// block number in x dimension
	int by = 0;											// block number in y dimension
	int index = 0;											// global index

	float in_sr[IN_SR_SIZE];									// for shift-register based spatial blocking

	// initialize the shift register
	#pragma unroll
	for (int i = 0; i < IN_SR_SIZE; i++)
	{
		in_sr[i] = 0.0f;
	}

	// main loop
	while (index != comp_exit)
	{
		index++;											// increment global index

		// shift the shift register by SSIZE
		#pragma unroll
		for (int i = 0; i < IN_SR_BASE; i++)
		{
			in_sr[i] = in_sr[i + SSIZE];
		}

		#pragma unroll										// unroll main loop by SSIZE
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - 1;							// global x position offset, adjusted for halo
			int gy = by + y - 1;							// global y position, adjusted for halo
			int real_x = gx + i;							// global x position
			int comp_offset_z = z - 1;						// global z position, will be out-of-bound for first and last iterations
			int comp_offset = gx + gy * nx + comp_offset_z * nx * ny; // index offset for computation, reading from power buffer, and writing back to destination
			int read_offset = comp_offset + nx * ny;			// index offset to read from memory
			int real_block_x = x + i - 1;						// local x position in block, adjusted for halo
			int real_block_y = y - 1;						// local x position in block, adjusted for halo

			float north, south, east, west, below, above, center, local_pwr;

			// read input values from memory
			// avoid out-of-bound indexes in x and y directions
			if (real_x >= 0 && gy >= 0 && real_x < nx && gy < ny)
			{
				// there is nothing to read on the bottommost plane
				if (comp_offset_z != nz - 1)
				{
					in_sr[IN_SR_BASE + i] = src[read_offset + i];// read new temperature values from memory
				}

				// there is nothing to compute on the first out-of-bound plane and the halos
				// conditions on x dimension have been avoided since they only save a few memory accesses and could cause access splitting
				if (comp_offset_z >= 0 && real_block_y >= 0 && real_block_y < BLOCK_Y - 2)
				{
					local_pwr = pwr_in[comp_offset + i];		// read new power values from memory
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
			if (real_x == nx - 1)							// rightmost column in matrix
			{
				east = center;								// fall back on center
			}
			else
			{
				east = in_sr[SR_OFF_E + i];					// read from shift register
			}

			// north neighbor
			if (gy == 0)									// topmost row
			{
				north = center;							// fall back on center
			}
			else
			{
				north = in_sr[SR_OFF_N + i];					// read from shift register
			}

			// south neighbor
			if (gy == ny - 1)								// bottommost row
			{
				south = center;							// fall back on center
			}
			else
			{
				south = in_sr[SR_OFF_S + i];					// read from shift register
			}

			// below neighbor, the condition is intentionally written like this to follow the behavior of the baseline implementation
			if (comp_offset_z == 0)							// bottommost plane
			{
				below = center;							// fall back on center
			}
			else
			{
				below = in_sr[SR_OFF_B + i];					// read from shift register
			}

			// above neighbor, the condition is intentionally written like this to follow the behavior of the baseline implementation
			if (comp_offset_z == nz - 1)						// topmost plane
			{
				above = center;							// fall back on center
			}
			else
			{
				above = in_sr[SR_OFF_A + i];					// read from shift register
			}

			// write output values to memory
			// the following condition is to avoid halos and going out of bounds in all axes
			if (comp_offset_z >= 0 && real_block_x >= 0 && real_block_y >= 0 && real_block_x < BLOCK_X - 2 * 1 && real_block_y < BLOCK_Y - 2 * 1 && real_x < nx && gy < ny)
			{
				dst[comp_offset + i] = center    * cc  +
						             north     * cn  + south    * cs +
						             east      * ce  + west     * cw +
						             above     * ct  + below    * cb +
						             local_pwr * sdc + AMB_TEMP * ct;
			}
		}

		// equivalent to x = (x + SSIZE) % BLOCK_X
		x = (x + SSIZE) & (BLOCK_X - 1);						// move one chunk forward and reset to zero if end of row was reached

		if (x == 0)										// if one row finished
		{
			// equivalent to y = (y + 1) % BLOCK_Y
			y = (y + 1) & (BLOCK_Y - 1);						// go to next row

			if (y == 0)									// if one plane finished
			{
				if (z == nz)								// if on last compute plane (compute traverses one more plane than memory read/write)
				{
					z = 0;								// reset plane number

					if (bx == last_col)						// border of extended input grid in x direction
					{
						bx = 0;							// reset block column
						by += BLOCK_Y - 2;					// go to next block in y direction, account for overlapping
					}
					else
					{
						bx += BLOCK_X - 2;					// go to next block in x direction, account for overlapping
					}
				}
				else
				{
					z++;									// go to next plane
				}
			}
		}
	}
}