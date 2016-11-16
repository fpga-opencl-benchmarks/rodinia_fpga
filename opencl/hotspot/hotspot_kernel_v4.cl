#include "hotspot_common.h"
#include "../common/opencl_kernel_common.h"

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))

#ifndef BSIZE
	#if defined(AOCL_BOARD_de5net_a7)
		#define BSIZE 256
	#endif
#endif

#ifndef SIMD
	#if defined(AOCL_BOARD_de5net_a7)
		#define SIMD 16
	#endif
#endif

#ifndef CU
	#if defined(AOCL_BOARD_de5net_a7)
		#define CU 1
	#endif
#endif

__attribute__((reqd_work_group_size(BSIZE, BSIZE, 1)))
__attribute__((num_simd_work_items(SIMD)))
__attribute__((num_compute_units(CU)))
__kernel void hotspot(__global float* restrict power,		// power input
                      __global float* restrict temp_src,	// temperature input/output
                      __global float* restrict temp_dst,	// temperature input/output
                               int             grid_cols,	// number of columns
                               int             grid_rows,	// number of rows
                               float           step_div_Cap,	// step/Cap
                               float           Rx_1, 
                               float           Ry_1, 
                               float           Rz_1)
{
	__local float cache[BSIZE][BSIZE];
	float current, right, right_1, right_2, left, left_1, left_2; // these extra registers are used to help the compiler properly understand and coalesce memory accesses to Block RAMs

	// group id
	int bx = get_group_id(0);
	int by = get_group_id(1);

	// thread id
	int tx = get_local_id(0);
	int ty = get_local_id(1);

	// total number of groups in each dimension
	int gx = get_num_groups(0);
	int gy = get_num_groups(1);
	

	// each block finally computes result for a small block
	// after N iterations. 
	// it is the non-overlapping small blocks that cover 
	// all the input data

	// calculate the small block size
	int small_block_rows = BSIZE - 2;
	int small_block_cols = BSIZE - 2;

	// calculate the boundary for the big block 
	// based on the boundary of its small block
	int blkYstart = small_block_rows * by - 1; // boundary of small block -1
	int blkXstart = small_block_cols * bx - 1; // boundary of small block -1
	int blkYend   = blkYstart + BSIZE - 1;
	int blkXend   = blkXstart + BSIZE - 1;

	// calculate the global thread coordination, not equal to global thread id
	int xidx = blkXstart + tx;
	int yidx = blkYstart + ty;

	int index = grid_cols * yidx + xidx;

	// load data if it is within the valid input range
	if(IN_RANGE(yidx, 0, grid_rows-1) && IN_RANGE(xidx, 0, grid_cols-1))
	{
		cache[ty][tx] = temp_src[index];  // load the temperature data from global memory to shared memory, row-wise load
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// effective range within this block that falls within 
	// the valid range of the input data
	// used to rule out computation outside the boundary.

	/*int validYmin = (by == 0) ? 1 : 0;						// first row of blocks
	int validYmax = (by == gy - 1) ? grid_rows%small_block_rows : BSIZE - 1;	// last row of blocks
	int validXmin = (bx == 0) ? 1 : 0;						// first column of blocks
	int validXmax = (bx == gx - 1) ? grid_cols%small_block_cols : BSIZE - 1;	// last column of blocks*/
	
	int last_block_rows = grid_rows%small_block_rows;				// number of valid rows in the bottommost blocks
	int last_block_cols = grid_cols%small_block_cols;				// number of valid columns in the rightmost blocks

	int N = ty - 1;
	int S = ty + 1;
	int W = tx - 1;
	int E = tx + 1;

	N = (N == 0 && by == 0) ? 1 : N;						// might go out of bounds only on topmost row of blocks
	//W = (W == 0 && bx == 0) ? 1 : W;						// might go out of bounds only on leftmost row of blocks
	S = (S > last_block_rows && by == gy - 1) ? last_block_rows : S;		// might go out of bounds only on bottommost row of blocks
	//E = (E > last_block_cols && bx == gx - 1) ? last_block_cols : E;		// might go out of bounds only on rightmost row of blocks

	// in the following assignments, we try to load both possible values for left and right neighbors
	// and choose the correct one at the end; this allows accesses to be coalesced when SIMD is used
	// the original implementation chooses the correct address instead of the value and hence
	// the compiler cannot coalesce accesses due to variable address (dynamic indexing)
	right_1 = cache[ty][tx + 1];							// assuming that we are not on rightmost column of the matrix
	left_1  = cache[ty][tx - 1];							// assuming that we are not on the leftmost column of the matrix
	right_2 = left_2 = current = cache[ty][tx];					// assuming that we are on the leftmost or rightmost column of the matrix
	right   = (E > last_block_cols && bx == gx - 1) ? right_2 : right_1;		// choose correct value for right neighbor
	left    = (W == 0 && bx == 0)                   ? left_2  : left_1;		// choose correct value for left neighbor

	if(tx != 0 && tx != BSIZE - 1 &&						// tx equal to 0 or BSIZE - 1 are always out of bounds for computation and are only used for memory load
           ty != 0 && ty != BSIZE - 1 &&						// ty equal to 0 or BSIZE - 1 are always out of bounds for computation and are only used for memory load
	   !(ty > last_block_rows && by == gy - 1) &&					// prevent out of bound computation in the bottommost blocks
	   !(tx > last_block_cols && bx == gx - 1))					// prevent out of bound computation in the rightmost blocks
	{
		temp_dst[index] = current + step_div_Cap *
		                  (power[index] +
		                  (cache[S][tx] + cache[N][tx] - 2.0f * current) * Ry_1 +
		                  (right + left - 2.0f * current) * Rx_1 +
		                  (AMB_TEMP - current) * Rz_1);
	}
}
