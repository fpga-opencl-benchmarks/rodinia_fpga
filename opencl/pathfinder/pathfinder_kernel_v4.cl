#include "pathfinder_common.h"

#define IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define HALO 1

#ifndef SSIZE
	#define SSIZE 16
#endif
	
#ifndef CU
	#define CU 1
#endif

#ifndef UNROLL
	#define UNROLL 1
#endif

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__attribute__((num_simd_work_items(SSIZE)))
__attribute__((num_compute_units(CU)))
__kernel void dynproc_kernel(         int           iteration,
                             __global int* restrict gpuWall,
                             __global int* restrict gpuSrc,
                             __global int* restrict gpuResults,
                                      int           cols,
                                      int           startStep,
                                      int           border)
{
	int bx = get_group_id(0);
	int tx = get_local_id(0);
	
	// local buffers
	__local int __attribute__((memory, numbanks(1), bankwidth(4 * SSIZE), doublepump)) prev[BSIZE];
	int result_out, result_in;

	// Each block finally computes result for a small block
	// after N iterations.
	// it is the non-overlapping small blocks that cover
	// all the input data
	
	// calculate the small block size.
	int small_block_cols = BSIZE - (iteration * HALO * 2);

	// calculate the boundary for the block according to
	// the boundary of its small block
	int blkX = (small_block_cols * bx) - border;
	int blkXmax = blkX + BSIZE - 1;

	// calculate the global thread coordination
	int xidx = blkX + tx;

	// effective range within this block that falls within
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > cols-1) ? BSIZE - 1 - (blkXmax - cols + 1) : BSIZE - 1;

	if(IN_RANGE(xidx, 0, cols - 1))
	{
		result_in = gpuSrc[xidx];
	}
	// this barrier is unnecessary; however, removing it decreases the number of simultaneous work-groups the compiler
	// allows, which results in noticeable performance reduction...
	barrier(CLK_LOCAL_MEM_FENCE);

	bool computed;
	#pragma unroll UNROLL
	for (int i = 0; i < iteration; i++)
	{
		// avoid putting prev in the following condition block for correct
		// access coalescing and minimum number of ports to the local buffer
		float result;
		if (i == 0)			// first iteration
		{
			result = result_in;	// read the data read from global memory and saved on-chip
		}
		else if (computed)		// valid computation
		{
			result = result_out;// read output from previous iteration
		}
		prev[tx] = result;
		barrier(CLK_LOCAL_MEM_FENCE);

		computed = false;
		int center     = prev[tx];
		int left_temp  = prev[tx - 1];
		int right_temp = prev[tx + 1];
		barrier(CLK_LOCAL_MEM_FENCE);

		int left   = (tx - 1 < validXmin) ? center : left_temp;
		int right  = (tx + 1 > validXmax) ? center : right_temp;
		
		if(IN_RANGE(tx, i + 1, BSIZE - i - 2) && IN_RANGE(tx, validXmin, validXmax))
		{
			computed = true;
			int index = cols * (startStep + i) + xidx;
			int shortest = MIN(left, center);
			result_out = MIN(shortest, right) + gpuWall[index];
		}

		if(i == iteration - 1)
		{
			// we are on the last iteration, and thus don't need to 
			// compute for the next step.
			break;
		}
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on "computed"
	if (computed)
	{
		gpuResults[xidx] = result_out;
	}
}