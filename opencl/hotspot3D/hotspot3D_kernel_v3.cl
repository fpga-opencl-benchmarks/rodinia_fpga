#include "hotspot3D_common.h"

#ifndef SSIZE
	#define SSIZE 4
#endif

__attribute__((max_global_work_dim(0)))
__kernel void hotspotOpt1(__global float* restrict pIn,
                          __global float* restrict tIn,
                          __global float* restrict tOut,
                                   float           sdc,
                                   int             nx,
                                   int             ny,
                                   int             nz,
                                   float           ce,
                                   float           cw, 
                                   float           cn,
                                   float           cs,
                                   float           ct,
                                   float           cb, 
                                   float           cc)
{
	for(int z = 0; z < nz; z++)
	{
		for(int y = 0; y < ny; y++)
		{
			#pragma unroll SSIZE
			for(int x = 0; x < nx; x++)
			{
				int index = x + y * nx + z * nx * ny;
				float c = tIn[index];

				float w = (x == 0)      ? c : tIn[index - 1];
				float e = (x == nx - 1) ? c : tIn[index + 1];
				float n = (y == 0)      ? c : tIn[index - nx];
				float s = (y == ny - 1) ? c : tIn[index + nx];
				float b = (z == 0)      ? c : tIn[index - nx * ny];
				float t = (z == nz - 1) ? c : tIn[index + nx * ny];

				tOut[index] = c * cc + n * cn + s * cs + e * ce + w * cw + t * ct + b * cb + sdc * pIn[index] + ct * AMB_TEMP;
			}
		}
	}
}
