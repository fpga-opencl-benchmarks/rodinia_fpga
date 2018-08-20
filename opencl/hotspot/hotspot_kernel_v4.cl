#include "hotspot_common.h"

#ifndef UNROLL
	#define UNROLL 1
#endif

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))

__attribute__((reqd_work_group_size(BLOCK_X, BLOCK_Y, 1)))
__attribute__((num_simd_work_items(SSIZE)))
__kernel void hotspot(int             comb_iter,		// either equal to pyramid_height or the remaining number of iterations
             __global float* restrict power,			// power input
             __global float* restrict temp_src,		// temperature input/output
             __global float* restrict temp_dst,		// temperature input/output
                      int             grid_cols,		// Col of grid
                      int             grid_rows,		// Row of grid
                      int             pyramid_height,	// number of combined iterations or degree of temporal parallelism
                      float           step_div_Cap,	// number of steps divided by capacitance
                      float           Rx_1, 
                      float           Ry_1, 
                      float           Rz_1,
                      int             small_block_rows,
                      int             small_block_cols)
{	
	__local float __attribute__((memory, numbanks(1), bankwidth(4 * SSIZE), doublepump)) temp_local[BLOCK_Y][BLOCK_X];
	float power_local, temp_in, temp_out;

	// group number in each dimension
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// thread number in each dimension
	int tx = get_local_id(0);
	int ty = get_local_id(1);
	
	// calculate the boundary for the block according to 
	// the boundary of its small block
	int blkXmin = small_block_cols * bx - pyramid_height;
	int blkXmax = blkXmin + BLOCK_X - 1;
	int blkYmin = small_block_rows * by - pyramid_height;
	int blkYmax = blkYmin + BLOCK_Y - 1;

	// calculate the global thread coordination
	int loadXidx = blkXmin + tx;
	int loadYidx = blkYmin + ty;

	// load data if it is within the valid input range
	int index = grid_cols * loadYidx + loadXidx;

	if(IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1))
	{
		temp_in = temp_src[index];		// Load the temperature data from global memory to private memory
		power_local = power[index];		// Load the power data from global memory to private memory
	}
	// this barrier is unnecessary; however, removing it decreases the number of simultaneous work-groups the compiler
	// allows, which results in noticeable performance reduction...
	barrier(CLK_LOCAL_MEM_FENCE);

	// effective range within this block that falls within 
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validXmin = (blkXmin <       0      ) ? pyramid_height : 0;
	int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_X - 1 - (blkXmax - grid_cols + 1) : BLOCK_X - 1;
	int validYmin = (blkYmin <       0      ) ? pyramid_height : 0;
	int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_Y - 1 - (blkYmax - grid_rows + 1) : BLOCK_Y - 1;

	bool computed;
	#pragma unroll UNROLL
	for (int i = 0; i < comb_iter; i++)
	{
		// avoid putting temp_local in the following condition block for correct
		// access coalescing and minimum number of ports to the local buffer
		float temp;
		if (i == 0)           // first iteration
		{
			temp = temp_in;  // read the data read from global memory and saved on-chip
		}
		else if (computed)    // valid computation
		{
			temp = temp_out; // read output from previous iteration
		}
		temp_local[ty][tx] = temp;
		barrier(CLK_LOCAL_MEM_FENCE);

		computed = false;
		float center     = temp_local[ty][tx];
		float east_temp  = temp_local[ty][tx + 1];
		float west_temp  = temp_local[ty][tx - 1];
		float north_temp = temp_local[ty - 1][tx];
		float south_temp = temp_local[ty + 1][tx];
		barrier(CLK_LOCAL_MEM_FENCE);

		// fall on boundary in neighbor is out of bounds
		float east  = (tx + 1 > validXmax) ? center : east_temp;
		float west  = (tx - 1 < validXmin) ? center : west_temp;
		float north = (ty - 1 < validYmin) ? center : north_temp;
		float south = (ty + 1 > validYmax) ? center : south_temp;
		
		if(IN_RANGE(tx, i + 1, BLOCK_X - i - 2) && IN_RANGE(ty, i + 1, BLOCK_Y - i - 2) && IN_RANGE(tx, validXmin, validXmax) && IN_RANGE(ty, validYmin, validYmax))
		{
			computed = true;
			temp_out = center + step_div_Cap * (power_local   + 
					 (south + north - 2.0f * center) * Ry_1 + 
					 (east  + west  - 2.0f * center) * Rx_1 + 
					 (AMB_TEMP - center) * Rz_1);
		}
		
		if(i == comb_iter - 1) // last combined iteration
			break;
	}

	// update the global memory
	// after the last comb_iter, only threads coordinated within the 
	// small block perform the calculation and switch on "computed"
	if (computed)
	{
		temp_dst[index]= temp_out;		
	}
}
