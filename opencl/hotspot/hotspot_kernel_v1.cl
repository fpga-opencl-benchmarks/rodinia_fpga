#include "hotspot_common.h"

__kernel void hotspot(__global float* restrict power,		//power input
                      __global float* restrict temp_src,	//temperature input/output
                      __global float* restrict temp_dst,	//temperature input/output
                      int                      grid_cols,	//Col of grid
                      int                      grid_rows,	//Row of grid
                      float                    Cap,		//Capacitance
                      float                    Rx, 
                      float                    Ry, 
                      float                    Rz, 
                      float                    step)
{
	float step_div_Cap = step/Cap;
	float Rx_1 = 1 / Rx;
	float Ry_1 = 1 / Ry;
	float Rz_1 = 1 / Rz;

	for (int r = 0; r < grid_rows; ++r)
	{        
		for (int c = 0; c < grid_cols; ++c)
		{
			int index = c + r * grid_cols;
			int offset_n = (r == grid_rows - 1) ? 0 : grid_cols;
			int offset_s = (r ==      0       ) ? 0 : -grid_cols;
			int offset_e = (c == grid_cols - 1) ? 0 : 1;
			int offset_w = (c ==      0       ) ? 0 : -1;

			float v = power[index] +
				(temp_src[index + offset_n] + temp_src[index + offset_s] - 2.0f * temp_src[index]) * Ry_1 + 
				(temp_src[index + offset_e] + temp_src[index + offset_w] - 2.0f * temp_src[index]) * Rx_1 +
				(AMB_TEMP - temp_src[index]) * Rz_1;

			float delta = step_div_Cap * v;

			temp_dst[index] = temp_src[index] + delta;
		}
	}
}
