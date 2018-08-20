#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#ifndef SSIZE
	#define SSIZE 4
#endif

__attribute__((max_global_work_dim(0)))
__kernel void dynproc_kernel (__global int* restrict wall,
                              __global int* restrict src,
                              __global int* restrict dst,
                                       int  cols,
                                       int  t)
{
	#pragma unroll SSIZE
	for(int n = 0; n < cols; n++)
	{
		int min = src[n];
		// the following two accesses could be out-of-bound
		// however, adding a condition to prevent them from going out-of-bound prevents the compiler from coalescing the accesses
		// this does not cause any trouble at run-time
		int right = src[n + 1];
		int left  = src[n - 1];

		if (n > 0)
		{
			min = MIN(min, left);
		}
		if (n < cols - 1)
		{
			min = MIN(min, right);
		}
		dst[n] = wall[(t + 1) * cols + n] + min;
	}
}
