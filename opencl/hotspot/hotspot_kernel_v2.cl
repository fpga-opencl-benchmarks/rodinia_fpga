#include "hotspot_common.h"

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))

__attribute__((reqd_work_group_size(BSIZE, BSIZE, 1)))
__attribute__((num_simd_work_items(SSIZE)))
__kernel void hotspot(int             iteration,		// number of iteration
             __global float* restrict power,			// power input
             __global float* restrict temp_src,		// temperature input/output
             __global float* restrict temp_dst,		// temperature input/output
                      int             grid_cols,		// Col of grid
                      int             grid_rows,		// Row of grid
                      int             border_cols,	// border offset 
                      int             border_rows,	// border offset
                      float           step_div_Cap,	// number of steps divided by capacitance
                      float           Rx_1, 
                      float           Ry_1, 
                      float           Rz_1,
                      int             small_block_rows,
                      int             small_block_cols)
{	
	local float temp_on_cuda[BSIZE][BSIZE];
	local float power_on_cuda[BSIZE][BSIZE];
	local float temp_t[BSIZE][BSIZE]; // saving temporary temperature result

	int bx = get_group_id(0);
	int by = get_group_id(1);

	int tx = get_local_id(0);
	int ty = get_local_id(1);

	// calculate the boundary for the block according to 
	// the boundary of its small block
	int blkY = small_block_rows*by-border_rows;
	int blkX = small_block_cols*bx-border_cols;
	int blkYmax = blkY+BSIZE-1;
	int blkXmax = blkX+BSIZE-1;

	// calculate the global thread coordination
	int yidx = blkY+ty;
	int xidx = blkX+tx;

	// load data if it is within the valid input range
	int loadYidx=yidx, loadXidx=xidx;
	int index = grid_cols*loadYidx+loadXidx;
			 
	if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1))
	{
		temp_on_cuda[ty][tx] = temp_src[index];	// Load the temperature data from global memory to shared memory
		power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// effective range within this block that falls within 
	// the valid range of the input data
	// used to rule out computation outside the boundary.
	int validYmin = (blkY < 0) ? -blkY : 0;
	int validYmax = (blkYmax > grid_rows-1) ? BSIZE-1-(blkYmax-grid_rows+1) : BSIZE-1;
	int validXmin = (blkX < 0) ? -blkX : 0;
	int validXmax = (blkXmax > grid_cols-1) ? BSIZE-1-(blkXmax-grid_cols+1) : BSIZE-1;

	int N = ty-1;
	int S = ty+1;
	int W = tx-1;
	int E = tx+1;

	N = (N < validYmin) ? validYmin : N;
	S = (S > validYmax) ? validYmax : S;
	W = (W < validXmin) ? validXmin : W;
	E = (E > validXmax) ? validXmax : E;

	bool computed;
	for (int i=0; i<iteration ; i++){ 
		computed = false;
		if(IN_RANGE(tx, i+1, BSIZE-i-2) && IN_RANGE(ty, i+1, BSIZE-i-2) && IN_RANGE(tx, validXmin, validXmax) && IN_RANGE(ty, validYmin, validYmax))
		{
			computed = true;
			temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] + 
						  (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0f * temp_on_cuda[ty][tx]) * Ry_1 + 
						  (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0f * temp_on_cuda[ty][tx]) * Rx_1 + 
						  (AMB_TEMP - temp_on_cuda[ty][tx]) * Rz_1);
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(i==iteration-1)
			break;
		if(computed)	 //Assign the computation range
			temp_on_cuda[ty][tx]= temp_t[ty][tx];
			
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// update the global memory
	// after the last iteration, only threads coordinated within the 
	// small block perform the calculation and switch on ``computed''
	if (computed){
		temp_dst[index]= temp_t[ty][tx];		
	}
}
