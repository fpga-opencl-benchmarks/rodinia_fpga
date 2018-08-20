#include "hotspot_common.h"

#ifndef SSIZE
	#define SSIZE 4
#endif

__attribute__((max_global_work_dim(0)))
__kernel void hotspot(__global float* restrict power,		//power input
                      __global float* restrict temp_src,	//temperature input/output
                      __global float* restrict temp_dst,	//temperature input/output
                      int                      grid_cols,	//Col of grid
                      int                      grid_rows,	//Row of grid
                      float                    step_div_Cap,// number of steps divided by capacitance
                      float                    Rx_1, 
                      float                    Ry_1, 
                      float                    Rz_1)
{
	for (int row = 0; row < grid_rows; ++row)
	{
		#pragma unroll SSIZE
		for (int col = 0; col < grid_cols; ++col)
		{
			int index = col + row * grid_cols;
			float c = temp_src[index];

			float n = (row == grid_rows - 1) ? c : temp_src[index + grid_cols];
			float s = (row ==      0       ) ? c : temp_src[index - grid_cols];
			float e = (col == grid_cols - 1) ? c : temp_src[index + 1];
			float w = (col ==      0       ) ? c : temp_src[index - 1];

			float v = power[index] +	(n + s - 2.0f * c) * Ry_1 + (e + w - 2.0f * c) * Rx_1 + (AMB_TEMP - c) * Rz_1;
			float delta = step_div_Cap * v;
			temp_dst[index] = c + delta;
		}
	}
}
