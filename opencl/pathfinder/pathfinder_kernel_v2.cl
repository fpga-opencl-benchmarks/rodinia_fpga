#include "pathfinder_common.h"

#define IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define HALO 1

#ifndef SSIZE
	#define SSIZE 16
#endif
	
#ifndef CU
	#define CU 2
#endif

#ifndef UNROLL
	#define UNROLL 1
#endif

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__attribute__((num_simd_work_items(SSIZE)))
__attribute__((num_compute_units(CU)))
__kernel void dynproc_kernel (         int           iteration,
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
	__local int prev[BSIZE];
	__local int result[BSIZE];

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
	
	int W = tx - 1;
	int E = tx + 1;

	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool isValid = IN_RANGE(tx, validXmin, validXmax);

	if(IN_RANGE(xidx, 0, cols - 1))
	{
		prev[tx] = gpuSrc[xidx];
	}
	
	barrier(CLK_LOCAL_MEM_FENCE);

	bool computed;
	#pragma unroll UNROLL
	for (int i = 0; i < iteration; i++)
	{
		computed = false;
		
		if( IN_RANGE(tx, i + 1, BSIZE - i - 2) && isValid )
		{
			computed = true;
			int left = prev[W];
			int up = prev[tx];
			int right = prev[E];
			int shortest = MIN(left, up);
			shortest = MIN(shortest, right);
			
			int index = cols * (startStep + i) + xidx;
			result[tx] = shortest + gpuWall[index];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(i == iteration - 1)
		{
			// we are on the last iteration, and thus don't need to 
			// compute for the next step.
			break;
		}

		if(computed)
		{
			//Assign the computation range
			prev[tx] = result[tx];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the
	// small block perform the calculation and switch on "computed"
	if (computed)
	{
		gpuResults[xidx] = result[tx];
	}
}