#include "hotspot3D_common.h"

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
			for(int x = 0; x < nx; x++)
			{
				int c = x + y * nx + z * nx * ny;

				int w = (x == 0) ? c : c - 1;
				int e = (x == nx - 1) ? c : c + 1;
				int n = (y == 0) ? c : c - nx;
				int s = (y == ny - 1) ? c : c + nx;
				int b = (z == 0) ? c : c - nx * ny;
				int t = (z == nz - 1) ? c : c + nx * ny;

				tOut[c] = tIn[c]*cc + tIn[n]*cn + tIn[s]*cs + tIn[e]*ce + tIn[w]*cw + tIn[t]*ct + tIn[b]*cb + sdc * pIn[c] + ct*AMB_TEMP;
			}
		}
	}
}
