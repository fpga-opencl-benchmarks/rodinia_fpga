#include "../common/opencl_kernel_common.h"

#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__kernel void dynproc_kernel (__global int* RESTRICT wall,
                              __global int* RESTRICT src,
                              __global int* RESTRICT dst,
                                       int  cols,
                                       int  t)
{
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
