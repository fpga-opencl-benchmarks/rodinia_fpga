#include "../common/opencl_kernel_common.h"
#define ABS(x)          (((x) > 0.0) ? (x) : (-(x)))
#define ETA             0.3  //eta value
#define MOMENTUM        0.3  //momentum value

#define FADD_LATENCY 8

#ifndef FOR_UNROLL
	#define FOR_UNROLL 1
#endif
#define FOR_REDUCTION_LATENCY ((FOR_UNROLL + (FOR_UNROLL % 2)) / 2) * FADD_LATENCY

#ifndef OUT_UNROLL
	#define OUT_UNROLL 1
#endif
#define OUT_REDUCTION_LATENCY ((OUT_UNROLL + (OUT_UNROLL % 2)) / 2) * FADD_LATENCY

#ifndef HID_UNROLL
	#define HID_UNROLL 1
#endif
#define HID_REDUCTION_LATENCY ((HID_UNROLL + (HID_UNROLL % 2)) / 2) * FADD_LATENCY

#ifndef ADJ_UNROLL
	#define ADJ_UNROLL 1
#endif

inline float squash(float x)
{
  return (1.0 / (1.0 + exp(-x)));
}

__attribute__((max_global_work_dim(0)))
__kernel void bpnn_layerforward(__global float* RESTRICT l1,
                                __global float* RESTRICT l2,
                                __global float* RESTRICT conn,
                                         int             n1,
                                         int             n2)
{
	float sum;
	int j, k, l;
	float shift_reg[FOR_REDUCTION_LATENCY + 1];

	// Set up thresholding unit
	l1[0] = 1.0;

	// For each unit in second layer
	for (j = 1; j <= n2; j++)
	{
		// initialize shift register
		#pragma unroll
		for (l = 0; l < FOR_REDUCTION_LATENCY + 1; l++)
		{
			shift_reg[l] = 0;
		}

		// Compute weighted sum of its inputs
		sum = 0.0;
		#pragma unroll FOR_UNROLL
		for (k = 0; k <= n1; k++)
		{
			shift_reg[FOR_REDUCTION_LATENCY] = shift_reg[0] + conn[k * (n2 + 1) + j] * l1[k];

			// shifting
			#pragma unroll
			for (l = 0; l < FOR_REDUCTION_LATENCY; l++)
			{
				shift_reg[l] = shift_reg[l + 1];
			}
		}

		//final reduction
		#pragma unroll
		for (l = 0; l < FOR_REDUCTION_LATENCY; l++)
		{
			sum += shift_reg[l];
		}
		l2[j] = squash(sum);
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void bpnn_output_error(__global float* RESTRICT delta,
                                __global float* RESTRICT target,
                                __global float* RESTRICT output,
                                         int             nj,
                                __global float* RESTRICT err)
{
	int j, l;
	float o, t, errsum;
	float shift_reg[OUT_REDUCTION_LATENCY + 1];

	// initialize shift register
	#pragma unroll
	for (l = 0; l < OUT_REDUCTION_LATENCY + 1; l++)
	{
		shift_reg[l] = 0;
	}

	errsum = 0.0;
	#pragma unroll OUT_UNROLL
	for (j = 1; j <= nj; j++)
	{
		o = output[j];
		t = target[j];
		delta[j] = o * (1.0 - o) * (t - o);

		shift_reg[OUT_REDUCTION_LATENCY] = shift_reg[0] + ABS(delta[j]);

		// shifting
		#pragma unroll
		for (l = 0; l < OUT_REDUCTION_LATENCY; l++)
		{
			shift_reg[l] = shift_reg[l + 1];
		}
	}

	//final reduction
	#pragma unroll
	for (l = 0; l < OUT_REDUCTION_LATENCY; l++)
	{
		errsum += shift_reg[l];
	}

	err[0] = errsum;
}

__attribute__((max_global_work_dim(0)))
__kernel void bpnn_hidden_error(__global float* RESTRICT delta_h,   
                                         int             nh, 
                                __global float* RESTRICT delta_o, 
                                         int             no, 
                                __global float* RESTRICT who, 
                                __global float* RESTRICT hidden, 
                                __global float* RESTRICT err)
{
	int j, k, l;
	float h, sum, errsum;
	float shift_reg[HID_REDUCTION_LATENCY + 1];

	errsum = 0.0;
	for (j = 1; j <= nh; j++)
	{
		// initialize shift register
		#pragma unroll
		for (l = 0; l < HID_REDUCTION_LATENCY + 1; l++)
		{
			shift_reg[l] = 0;
		}

		h = hidden[j];
		sum = 0.0;
		#pragma unroll HID_UNROLL
		for (k = 1; k <= no; k++)
		{
			shift_reg[HID_REDUCTION_LATENCY] = shift_reg[0] + delta_o[k] * who[j * (no + 1) + k];

			// shifting
			#pragma unroll
			for (l = 0; l < HID_REDUCTION_LATENCY; l++)
			{
				shift_reg[l] = shift_reg[l + 1];
			}
		}

		//final reduction
		#pragma unroll
		for (l = 0; l < HID_REDUCTION_LATENCY; l++)
		{
			sum += shift_reg[l];
		}

		delta_h[j] = h * (1.0 - h) * sum;
		errsum += ABS(delta_h[j]);
	}
	err[0] = errsum;
}

__attribute__((max_global_work_dim(0)))
__kernel void bpnn_adjust_weights(__global float* RESTRICT delta,
                                           int             ndelta,
                                  __global float* RESTRICT ly,
                                           int             nly,
                                  __global float* RESTRICT w,
                                  __global float* RESTRICT oldw)
{
	float new_dw;
	int k, j;

	ly[0] = 1.0;
	#pragma ivdep
	for (j = 1; j <= ndelta; j++)
	{
		#pragma ivdep
		#pragma unroll ADJ_UNROLL
		for (k = 0; k <= nly; k++)
		{
			new_dw = ((ETA * delta[j] * ly[k]) + (MOMENTUM * oldw[k * (ndelta + 1) + j]));
			w[k * (ndelta + 1) + j] += new_dw;
			oldw[k * (ndelta + 1) + j] = new_dw;
		}
	}
}