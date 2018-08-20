#include "pathfinder_common.h"

#define MIN(a, b) ((a)<=(b) ? (a) : (b))

// input shift register parameters
#define IN_SR_BASE		BSIZE + 1				// this shows the point to write into the shift register; one row to reach north neighbor and one extra cell for north_west
#define IN_SR_SIZE		IN_SR_BASE + SSIZE		// SSIZE extra indexes to support vectorization

// offsets for reading from the input shift register
#define SR_OFF_N		1					// north
#define SR_OFF_NE		2					// north_east
#define SR_OFF_NW		0					// north_west

#define HALO_SIZE		rem_rows				// the dependency pattern forms a cone with the same base size and height

__attribute__((max_global_work_dim(0)))
__kernel void dynproc_kernel (__global int* restrict wall,
                              __global int* restrict src,
						__global int* restrict dst,
                                       int           cols,
                                       int           rem_rows,
                                       int           starting_row,
                                       int           comp_exit)
{
	int shift_reg[IN_SR_SIZE];								// shift register for spatial blocking

	// initialize the shift register
	#pragma unroll
	for (int i = 0; i < IN_SR_SIZE; i++)
	{
		shift_reg[i] = 0.0f;
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
			shift_reg[i] = shift_reg[i + SSIZE];
		}

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;						// global x position offset
			int real_x = gx + i;							// global x position
			int real_y = y + starting_row;					// global y position
			int read_offset = real_x + (real_y - 1) * cols;		// index to read from memory, wall input starts from second row of the input grid
			int real_block_x = x + i - HALO_SIZE;				// local x position in block, adjusted for halo

			int north = 0x7ffffff, north_east = 0x7ffffff, north_west = 0x7ffffff, center = 0x7ffffff; // initialize to biggest positive integer
			int min = 0;

			// this read is done here since if it is merged into the next condition,
			// the compiler will refuse to coalesce the memory accesses
			// only required in first block row, out-of-bound reads are also skipped
			int in = (y == 0 && real_x >= 0 && real_x < cols) ? src[real_x] : 0;

			// the following condition is to avoid halos and going out of bounds in all axes
			// since the dependency pattern forms a cone, y gets involved in this case
			if (real_block_x >= y - HALO_SIZE && real_block_x < BSIZE - 2 * y && real_x >= 0 && real_x < cols)
			{
				center = (y == 0) ? in : wall[read_offset];
			}

			if (y > 0)									// avoid first row in block since to computation is done
			{
				north = shift_reg[SR_OFF_N + i];
				if (real_x > 0)							// avoid first column
				{
					north_west = shift_reg[SR_OFF_NW + i];
				}
				if (real_x < cols - 1)						// avoid last column
				{
					north_east = shift_reg[SR_OFF_NE + i];
				}
				min = MIN(north, MIN(north_east, north_west));
			}

			// write output values to shift register or memory
			// the following condition is to avoid halos and going out of bounds in all axes
			// since the dependency pattern forms a cone, y gets involved in this case
			if (real_y >= 0 && real_block_x >= y - HALO_SIZE && real_block_x < BSIZE - 2 * y && real_x >= 0 && real_x < cols)
			{
				shift_reg[IN_SR_BASE + i] = center + min;
				if (y == rem_rows)
				{
					dst[real_x] = shift_reg[IN_SR_BASE + i];
				}
				//if (y > 0)
					//printf("row: %04d, col: %04d, center: %d, N: %d, NE: %d, NW: %d, min: %d, out: %d\n", real_y, real_x, center, north, north_east, north_west, min, shift_reg[IN_SR_BASE + i]);
			}
		}

		x = (x + SSIZE) & (BSIZE - 1);						// move one chunk forward and reset to zero if end of row was reached, alternative for x = (x + SSIZE) % BSIZE

		if (x == 0)										// if one row finished
		{
			y++;											// go to next row

			if (y == rem_rows + 1)							// if on last row in block, traverses one more row than it should for write-back
			{
				y = 0;									// reset row number
				bx += BSIZE - 2 * HALO_SIZE;					// go to next block, account for overlapping
			}
		}
	}
}