#include "../common/opencl_kernel_common.h"

#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#ifndef UNROLL
	#define UNROLL 4
#endif

__attribute__((max_global_work_dim(0)))
__kernel void dynproc_kernel (__global int* RESTRICT wall,
                              __global int* RESTRICT src,
                              __global int* RESTRICT dst,
                                       int  cols,
                                       int  t)
{
	#pragma unroll UNROLL
	for(int n = 0; n < cols; n++)
	{
		int min = src[n];
		if (n > 0)
		{
			min = MIN(min, src[n - 1]);
		}
		if (n < cols-1)
		{
			min = MIN(min, src[n + 1]);
		}
		dst[n] = wall[(t + 1) * cols + n] + min;
	}

}
