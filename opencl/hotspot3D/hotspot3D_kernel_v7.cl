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

#include "hotspot3D_common.h"

#ifndef CSIZE
	#define CSIZE 16
#endif

typedef struct
{
	float data[SSIZE];
} CHAN_WIDTH;

typedef struct
{
	int data[6];
} INT6;

// input shift register parameters
#define IN_SR_BASE		2 * RAD * BLOCK_X * BLOCK_Y		// this shows the point to write into the shift register; RAD rows for top neighbors, one row for center, and (RAD - 1) for bottom
#define IN_SR_SIZE		IN_SR_BASE + SSIZE				// SSIZE indexes are enough for the bottommost row

// power shift register parameters
#define PWR_SR_BASE		RAD * BLOCK_X * BLOCK_Y			// this shows the point to write into the shift register; one row for center cell, and (RAD - 1) for bottom neighbors
#define PWR_SR_SIZE		PWR_SR_BASE + SSIZE				// SSIZE indexes are enough for the bottommost row

// offsets for reading from the input shift register
#define SR_OFF_C		RAD * BLOCK_X * BLOCK_Y			// center
#define SR_OFF_S		SR_OFF_C + BLOCK_X				// south  (-y direction)
#define SR_OFF_N		SR_OFF_C - BLOCK_X				// north  (+y direction)
#define SR_OFF_E		SR_OFF_C + 1					// east   (-x direction)
#define SR_OFF_W		SR_OFF_C - 1					// west   (+x direction)
#define SR_OFF_A		SR_OFF_C + BLOCK_X * BLOCK_Y		// above  (-z direction), this is intentionally written like this to follow the behavior of the baseline implementation
#define SR_OFF_B		SR_OFF_C - BLOCK_X * BLOCK_Y		// below  (+z direction), this is intentionally written like this to follow the behavior of the baseline implementation

#pragma OPENCL EXTENSION cl_altera_channels : enable;

channel CHAN_WIDTH in_ch[TIME]            __attribute__((depth(CSIZE)));
channel CHAN_WIDTH pwr_ch[TIME]           __attribute__((depth(CSIZE)));
channel CHAN_WIDTH out_ch                 __attribute__((depth(CSIZE)));
channel float8     const_fl_ch[TIME + 1]  __attribute__((depth(0)));
channel INT6       const_int_ch[TIME + 1] __attribute__((depth(0)));

__attribute__((max_global_work_dim(0)))
__kernel void read(__global const float* restrict pwr_in,			// power input
                   __global const float* restrict src,				// temperature input
                            const float           sdc_,			// step/Capacitance
                            const int             nx_,				// x dimension size
                            const int             ny_,				// y dimension size
                            const int             nz_,				// z dimension size
                            const float           ce_,
                            const float           cw_, 
                            const float           cn_,
                            const float           cs_,
                            const float           ct_,
                            const float           cb_, 
                            const float           cc_,
                            const int             last_col_,		// exit condition for in x direction
                            const int             comp_exit_,		// exit condition for compute loop
                            const int             mem_exit,			// exit condition for memory loop
                            const int             rem_iter_)		// remaining iterations
{
	// ugly work-around to prevent the stupid compiler from inferring ultra-deep channels
	// for passing the constant values which wastes a lot of Block RAMs.
	// this involves creating a false cycle of channels and passing the values through all
	// the autorun kernels and back to this kernel; this disables the compiler's channel depth optimization.
	const float8 constants1_ = (float8)(sdc_, ce_, cw_, cn_, cs_, ct_, cb_, cc_);

	INT6 constants2_;
	constants2_.data[0] = nx_;
	constants2_.data[1] = ny_;
	constants2_.data[2] = nz_;
	constants2_.data[3] = last_col_;
	constants2_.data[4] = comp_exit_;
	constants2_.data[5] = rem_iter_;

	write_channel_altera(const_fl_ch[0] , constants1_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel_altera(const_int_ch[0], constants2_);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const float8 constants1 = read_channel_altera(const_fl_ch[TIME]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const INT6 constants2 = read_channel_altera(const_int_ch[TIME]);

	const float sdc = constants1.s0;
	const float ce  = constants1.s1;
	const float cw  = constants1.s2;
	const float cn  = constants1.s3;
	const float cs  = constants1.s4;
	const float ct  = constants1.s5;
	const float cb  = constants1.s6;
	const float cc  = constants1.s7;

	const int nx = constants2.data[0];
	const int ny = constants2.data[1];
	const int nz = constants2.data[2];
	const int last_col  = constants2.data[3];
	const int comp_exit = constants2.data[4];
	const int rem_iter  = constants2.data[5];

	// starting points
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;											// block number in x dimension
	int by = 0;											// block number in y dimension
	int index = 0;											// global index
	
	while (index != mem_exit)
	{
		index++;											// increment global index
		
		CHAN_WIDTH in, pwr;

		// read data from memory
		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;						// global x position offset, adjusted for halo
			int gy = by + y - HALO_SIZE;						// global y position, adjusted for halo
			int real_x = gx + i;							// global x position
			int read_offset = gx + gy * nx + z * nx * ny;		// index offset to read from memory

			// input value
			if (real_x >= 0 && gy >= 0 && real_x < nx && gy < ny) // avoid out-of-bound indexes in x and y directions; there is also nothing to read on the bottommost row
			{
				in.data[i] = src[PAD + read_offset + i];		// read new temperature values from memory
				pwr.data[i] = pwr_in[PAD + read_offset + i];		// read new power values from memory
			}
		}

		write_channel_altera(in_ch[0], in);					// write input values to channel as a vector
		write_channel_altera(pwr_ch[0], pwr);					// write input values to channel as a vector

		// equivalent to x = (x + SSIZE) % BLOCK_X
		x = (x + SSIZE) & (BLOCK_X - 1);						// move one chunk forward and reset to zero if end of row was reached

		if (x == 0)										// if one row finished
		{
			// equivalent to y = (y + 1) % BLOCK_Y
			y = (y + 1) & (BLOCK_Y - 1);						// go to next row

			if (y == 0)									// in one plane finished
			{
				if (z == nz - 1)							// if on last plane
				{
					z = 0;								// reset plane number

					if (bx == last_col)						// border of extended input grid in x direction
					{
						bx = 0;							// reset block column
						by += BLOCK_Y - BACK_OFF;			// go to next block in y direction, account for overlapping
					}
					else
					{
						bx += BLOCK_X - BACK_OFF;			// go to next block in x direction, account for overlapping
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

__attribute__((max_global_work_dim(0)))
__attribute__((autorun))
__attribute__((num_compute_units(TIME,1,1)))
__kernel void compute()
{
	const int ID = get_compute_id(0);

	const float8 constants1 = read_channel_altera(const_fl_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	const INT6  constants2 = read_channel_altera(const_int_ch[ID]);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel_altera(const_fl_ch[ID + 1] , constants1);
	mem_fence(CLK_CHANNEL_MEM_FENCE);
	write_channel_altera(const_int_ch[ID + 1], constants2);

	const float sdc = constants1.s0;
	const float ce  = constants1.s1;
	const float cw  = constants1.s2;
	const float cn  = constants1.s3;
	const float cs  = constants1.s4;
	const float ct  = constants1.s5;
	const float cb  = constants1.s6;
	const float cc  = constants1.s7;

	const int nx = constants2.data[0];
	const int ny = constants2.data[1];
	const int nz = constants2.data[2];
	const int last_col  = constants2.data[3];
	const int comp_exit = constants2.data[4];
	const int rem_iter  = constants2.data[5];

	float in_sr[IN_SR_SIZE];									// shift register for spatial blocking
	float pwr_sr[PWR_SR_SIZE];								// this is needed to hold one plane + SSIZE values since each time step is RAD planes behind the previous one

	// initialize shift registers
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

	// starting points
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;											// block number in x dimension
	int by = 0;											// block number in y dimension
	int index = 0;											// global index
	
	while (index != comp_exit)
	{
		index++;											// increment global index

		int comp_offset_z = z - RAD;							// global z position, will be out-of-bound for first and last RAD iterations

		CHAN_WIDTH in, pwr, out, curr_pwr;
		
		// shift
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
		if (comp_offset_z != nz - 1)							// nothing to read on last plane
		{
			in = read_channel_altera(in_ch[ID]);				// read input values from channel as a vector from previous time step (or read kernel for ID == 0)
			pwr = read_channel_altera(pwr_ch[ID]);				// read power values from channel as a vector from read kernel
		}

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;						// global x position offset, adjusted for halo
			int gy = by + y - HALO_SIZE;						// global y position, adjusted for halo
			int real_x = gx + i;							// global x position

			float north, south, east, west, below, above, center, local_pwr;
			
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

			// write output values as array elements
			if (ID < rem_iter)								// if iteration is not out of bound
			{
				out.data[i] = center    * cc  +
						    north     * cn  + south    * cs +
						    east      * ce  + west     * cw +
						    above     * ct  + below    * cb +
						    local_pwr * sdc + AMB_TEMP * ct;
			}
			else
			{
				out.data[i] = center;						// pass input data directly to output
			}

			//if (ID == 0 && gy >= 0 && real_x >= 0 && gy < 1 && real_x < 10 && z == 1)
			//if (real_x + gy * nx + comp_offset_z * nx * ny == 0)
				//printf("c*cc: %0.12f, n*cn: %0.12f, s*cs: %0.12f, e*ce: %0.12f, w*cw: %0.12f, a*ca: %0.12f, b*cb: %0.12f, sdc*p: %0.12f, ct*amb: %0.12f, out: %0.12f\n", center * cc, north * cn, south* cs, east * ce, west * cw, above * ct, below * cb, local_pwr * sdc, AMB_TEMP * ct, out.data[i]);
				//printf("x: %02d, y: %02d, z: %d, index: %04d, center: %f, north: %f, south: %f, left: %f, right: %f, above: %f, below: %f, pow: %f, out: %f\n", real_x, gy, comp_offset_z, real_x + gy * nx + comp_offset_z * nx * ny, center, north, south, west, east, above, below, local_pwr, out.data[i]);
		}

		// write output values
		if (comp_offset_z >= 0)								// nothing to write on first plane
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
						by += BLOCK_Y - BACK_OFF;			// go to next block in y direction, account for overlapping
					}
					else
					{
						bx += BLOCK_X - BACK_OFF;			// go to next block in x direction, account for overlapping
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

__attribute__((max_global_work_dim(0)))
__kernel void write(__global       float* restrict dst,			// temperature output
                             const int             nx,				// x dimension size
                             const int             ny,				// y dimension size
                             const int             nz,				// z dimension size
                             const int             last_col,		// exit condition for in x direction
                             const int             mem_exit)		// exit condition for memory loop
{
	// starting points
	int x = 0;
	int y = 0;
	int z = 0;
	int bx = 0;
	int by = 0;
	int index = 0;
	
	while (index != mem_exit)
	{
		index++;											// increment global index

		int gx = bx + x - HALO_SIZE;							// global x position, adjusted for halo
		int gy = by + y - HALO_SIZE;							// global y position, adjusted for halo
		
		CHAN_WIDTH out;

		out = read_channel_altera(out_ch);						// read output values from channel as a vector

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int gx = bx + x - HALO_SIZE;						// global x position offset, adjusted for halo
			int gy = by + y - HALO_SIZE;						// global y position, adjusted for halo
			int write_offset = gx + gy * nx + z * nx * ny;		// index offset to write to memory
			int real_block_x = x + i - HALO_SIZE;				// local x position in block, adjusted for halo
			int real_block_y = y - HALO_SIZE;					// local x position in block, adjusted for halo
			int real_x = gx + i;							// global x position

			// the following condition is to avoid halos and going out of bounds in all axes
			if (real_block_x >= 0 && real_block_y >= 0 && real_block_x < BLOCK_X - 2 * HALO_SIZE && real_block_y < BLOCK_Y - 2 * HALO_SIZE && real_x < nx && gy < ny)
			{
				dst[PAD + write_offset + i] = out.data[i];
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
				if (z == nz - 1)							// if on last compute plane
				{
					z = 0;								// reset plane number

					if (bx == last_col)						// border of extended input grid in x direction
					{
						bx = 0;							// reset block column
						by += BLOCK_Y - BACK_OFF;			// go to next block in y direction, account for overlapping
					}
					else
					{
						bx += BLOCK_X - BACK_OFF;			// go to next block in x direction, account for overlapping
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