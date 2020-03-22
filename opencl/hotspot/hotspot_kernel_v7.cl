//============================================================================================================
// (c) 2019, Hamid Reza Zohouri @ Tokyo Institute of Technology
//
// Using, modifying, and distributing this kernel file is permitted for educational, research, and non-profit
// use cases, as long as this copyright block is kept intact. Using this kernel file in any shape or form,
// including using it a template/skeleton to develop similar code, is forbidden for commercial/for-profit
// purposes, except with explicit permission from the author (Hamid Reza Zohouri).
//
// Contact point: https://www.linkedin.com/in/hamid-reza-zohouri-9aa00230/
//=============================================================================================================

#include "hotspot_common.h"

#ifndef CSIZE
	#define CSIZE 16
#endif

typedef struct
{
	float data[SSIZE];
} CHAN_WIDTH;

// input shift register parameters
#define IN_SR_BASE		2 * RAD * BSIZE		// this shows the point to write into the shift register; RAD rows for top neighbors, one row for center, and (RAD - 1) for bottom
#define IN_SR_SIZE		IN_SR_BASE + SSIZE		// SSIZE indexes are enough for the bottommost row

// power shift register parameters
#define PWR_SR_BASE		RAD * BSIZE			// this shows the point to write into the shift register; RAD rows for top neighbors, one row for center, and (RAD - 1) for bottom
#define PWR_SR_SIZE		PWR_SR_BASE + SSIZE		// SSIZE indexes are enough for the bottommost row

// offsets for reading from the input shift register
#define SR_OFF_C		RAD * BSIZE			// center
#define SR_OFF_N		SR_OFF_C + RAD * BSIZE	// north
#define SR_OFF_S		SR_OFF_C - RAD * BSIZE	// south
#define SR_OFF_E		SR_OFF_C + RAD			// east
#define SR_OFF_W		SR_OFF_C - RAD			// west

#pragma OPENCL EXTENSION cl_altera_channels : enable;

channel CHAN_WIDTH in_ch[TIME]            __attribute__((depth(CSIZE)));
channel CHAN_WIDTH pwr_ch[TIME]           __attribute__((depth(CSIZE)));
channel CHAN_WIDTH out_ch                 __attribute__((depth(CSIZE)));
channel float4     const_fl_ch[TIME + 1]  __attribute__((depth(0)));
channel int4       const_int_ch[TIME + 1] __attribute__((depth(0)));

__attribute__((max_global_work_dim(0)))
__kernel void read(__global const float* restrict src,				// temperature input
                   __global const float* restrict pwr_in,			// power input
                            const int             grid_cols_,		// number of columns
                            const int             grid_rows_,		// number of rows
                            const float           sdc_,			// step/Capacitance
                            const float           Rx_1_,
                            const float           Ry_1_,
                            const float           Rz_1_,
                            const int             comp_exit_,		// exit condition for compute loop
                            const int             mem_exit,			// exit condition for memory loop
                            const int             rem_iter_)		// remaining iterations
{
	// ugly work-around to prevent the stupid compiler from inferring ultra-deep channels
	// for passing the constant values which wastes a lot of Block RAMs.
	// this involves creating a false cycle of channels and passing the values through all
	// the autorun kernels and back to this kernel; this disables the compiler's channel depth optimization.
	float4 constants1_ = (float4)(sdc_, Rx_1_, Ry_1_, Rz_1_);;
	int4   constants2_ = (int4)(grid_rows_, grid_cols_, comp_exit_, rem_iter_);

	write_channel_altera(const_fl_ch[0] , constants1_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel_altera(const_int_ch[0], constants2_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const float4 constants1 = read_channel_altera(const_fl_ch[TIME]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const int4 constants2 = read_channel_altera(const_int_ch[TIME]);

	const float sdc  = constants1.s0;
	const float Rx_1 = constants1.s1;
	const float Ry_1 = constants1.s2;
	const float Rz_1 = constants1.s3;

	const int grid_rows = constants2.s0;
	const int grid_cols = constants2.s1;
	const int comp_exit = constants2.s2;
	const int rem_iter  = constants2.s3;

	// starting point
	int x = 0;
	int y = 0;
	int bx = 0;
	int index = 0;
	
	while (index != mem_exit)
	{
		index++;											// increment global index

		CHAN_WIDTH in, pwr;

		// read data from memory
		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;						// global x position offset, adjusted for halo
			int real_x = gx + i;							// global x position
			int read_offset = gx + y * grid_cols;				// index to read from memory

			// input and power value
			if (real_x >= 0 && real_x < grid_cols)				// avoid out-of-bound indexes in x direction
			{
				in.data[i]  = src[PAD + read_offset + i];		// read new values from memory
				pwr.data[i] = pwr_in[PAD + read_offset + i];		// read new values from memory
			}
		}

		write_channel_altera(in_ch[0], in);					// write input values to channel as a vector
		write_channel_altera(pwr_ch[0], pwr);					// write power values to channel as a vector

		// equivalent to x = (x + SSIZE) % BSIZE
		x = (x + SSIZE) & (BSIZE - 1);						// move one chunk forward and reset to zero if end of row was reached

		if (x == 0)										// if one row finished
		{
			if (y == grid_rows - 1)							// if on last row
			{
				y = 0;									// reset row number
				bx += BSIZE - BACK_OFF;						// go to next block, account for overlapping
			}
			else
			{
				y++;										// go to next row
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(TIME,1,1)))
__kernel void compute()
{
	const int ID = get_compute_id(0);

	const float4 constants1 = read_channel_altera(const_fl_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const int4  constants2 = read_channel_altera(const_int_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel_altera(const_fl_ch[ID + 1] , constants1);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel_altera(const_int_ch[ID + 1], constants2);

	const float sdc  = constants1.s0;
	const float Rx_1 = constants1.s1;
	const float Ry_1 = constants1.s2;
	const float Rz_1 = constants1.s3;

	const int grid_rows = constants2.s0;
	const int grid_cols = constants2.s1;
	const int comp_exit = constants2.s2;
	const int rem_iter  = constants2.s3;

	float in_sr[IN_SR_SIZE];									// shift register for spatial blocking
	float pwr_sr[PWR_SR_SIZE];								// this shift register is needed to hold one row + SSIZE values since each time step is one row behind the previous one

	// initialize
	#pragma unroll
	for (int i = 0; i < IN_SR_SIZE; i++)
	{
		in_sr[i] = 0.0f;
	}

	#pragma unroll
	for (int i = 0; i < PWR_SR_SIZE; i++)
	{
		pwr_sr[i] = 0.0f;
	}

	// starting point
	int x = 0;
	int y = 0;
	int bx = 0;
	int index = 0;

	while (index != comp_exit)
	{
		index++;											// increment global index

		int comp_offset_y = y - RAD;							// global y position, will be out-of-bound for first and last iterations

		CHAN_WIDTH in, pwr, out, curr_pwr;
		
		// shift shift register by SSIZE
		#pragma unroll
		for (int i = 0; i < IN_SR_BASE; i++)
		{
			in_sr[i] = in_sr[i + SSIZE];
		}
		#pragma unroll
		for (int i = 0; i < PWR_SR_BASE; i++)
		{
			pwr_sr[i] = pwr_sr[i + SSIZE];
		}

		// read input and power values
		if (comp_offset_y != grid_rows - 1)					// nothing to read on last row
		{
			in = read_channel_altera(in_ch[ID]);				// read input values from channel as a vector from previous time step (or read kernel for ID == 0)
			pwr = read_channel_altera(pwr_ch[ID]);				// read power values from channel as a vector from previous time step
		}

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;						// global x position offset
			int real_x = gx + i;							// global x position

			float north, south, east, west, center, local_pwr;
			
			in_sr[IN_SR_BASE + i] = in.data[i];				// read input values as array elements
			pwr_sr[PWR_SR_BASE + i] = pwr.data[i];				// read power values as array elements and write to shift register
			local_pwr = curr_pwr.data[i] = pwr_sr[i];			// read center power value from previous row in the shift register

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

			// write output values as array elements
			if (ID < rem_iter)								// if iteration is not out of bound
			{
				out.data[i] = center + delta;					// write output values as array elements
			}
			else
			{
				out.data[i] = center;						// pass input data directly to output
			}
			//if (ID == 1 && comp_offset_y >=0 && comp_offset_y < grid_rows && real_x >=0 && real_x < grid_cols)
				//printf("ID: %d, row: %04d, col: %04d, center: %f, top: %f, bottom: %f, left: %f, right: %f, pow: %f, out: %f\n", ID, comp_offset_y, real_x, center, north, south, west, east, local_pwr, out.data[i]);
		}

		// write output values
		if (comp_offset_y >= 0)								// nothing to write on first row
		{
			if (ID == TIME - 1)								// only if last time step
			{
				write_channel_altera(out_ch, out);				// write output values to channel as a vector for write back to memory
			}
			else											// avoid creating the following channel if the next time step "doesn't exist"
			{
				write_channel_altera(in_ch[ID + 1], out);		// write output values to channel as a vector for next time step
				write_channel_altera(pwr_ch[ID + 1], curr_pwr);	// forward power values to next time step as a vector, use top row of pwr_sr
			}
		}

		// equivalent to x = (x + SSIZE) % BSIZE
		x = (x + SSIZE) & (BSIZE - 1);						// move one chunk forward and reset to zero if end of row was reached

		if (x == 0)										// if one row finished
		{
			if (y == grid_rows)								// if on last row (compute traverses one more plane than memory read/write)
			{
				y = 0;									// reset row number
				bx += BSIZE - BACK_OFF;						// go to next block, account for overlapping
			}
			else
			{
				y++;										// go to next row
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void write(__global       float* restrict dst,			// temperature output
                             const int             grid_cols,		// number of columns
                             const int             grid_rows,		// number of rows
                             const int             mem_exit)		// loop exit condition
{
	// starting point
	int x = 0;
	int y = 0;
	int bx = 0;
	int index = 0;

	while (index != mem_exit)
	{
		index++;											// increment global index

		CHAN_WIDTH out;

		out = read_channel_altera(out_ch);						// read output values from channel as a vector

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;						// global x position, adjusted for halo
			int write_offset = gx + y * grid_cols;				// global index
			int real_block_x = x + i - HALO_SIZE;				// local x position in block, adjusted for halo
			int real_x = gx + i;							// global x position

			// the following condition is to avoid halos and going out of bounds in either axes
			if (real_block_x >= 0 && real_block_x < BSIZE - 2 * HALO_SIZE && real_x < grid_cols)
			{
				dst[PAD + write_offset + i] = out.data[i];
			}
		}

		// equivalent to x = (x + SSIZE) % BSIZE
		x = (x + SSIZE) & (BSIZE - 1);						// move one chunk forward and reset to zero if end of row was reached

		if (x == 0)										// if one row finished
		{
			if (y == grid_rows - 1)							// if on last row
			{
				y = 0;									// reset row number
				bx += BSIZE - BACK_OFF;						// go to next block, account for overlapping
			}
			else
			{
				y++;										// go to next row
			}
		}
	}
}
