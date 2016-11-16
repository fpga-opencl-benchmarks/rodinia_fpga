#include "../common/opencl_kernel_common.h"

#define IN_RANGE(x, min, max) ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define BLOCK_SIZE 256
#define HALO 1

__kernel void dynproc_kernel (__global int* RESTRICT gpuWall,
                              __global int* RESTRICT gpuSrc,
                              __global int* RESTRICT gpuDst,
                                       int  cols,
                                       int  rows)
{
	for (int t = 0; t < rows - 1; t++)
	{
		int min;
		for(int n = 0; n < cols; n++)
		{
			min = gpuSrc[n];
			if (n > 0)
			{
				min = MIN(min, gpuSrc[n - 1]);
			}
			if (n < cols - 1)
			{
				min = MIN(min, gpuSrc[n + 1]);
			}
			gpuDst[n] = gpuWall[cols * t + n] + min;
			//printf("startStep: %02d, n: %02d, gpuSrc[n]: %02d, gpuDst[n]: %02d, min: %02d\n", t, n, gpuSrc[n], gpuDst[n], min);
		}
		
		if (t != rows - 2)
		{
			t++;
		}
		else
		{
			break;
		}
		
		for(int n = 0; n < cols; n++)
		{
			min = gpuDst[n];
			if (n > 0)
			{
				min = MIN(min, gpuDst[n - 1]);
			}
			if (n < cols - 1)
			{
				min = MIN(min, gpuDst[n + 1]);
			}
			gpuSrc[n] = gpuWall[cols * t + n] + min;
			//printf("startStep: %02d, n: %02d, gpuSrc[n]: %02d, gpuDst[n]: %02d, min: %02d\n", t, n, gpuDst[n], gpuSrc[n], min);
		}
	}
}




