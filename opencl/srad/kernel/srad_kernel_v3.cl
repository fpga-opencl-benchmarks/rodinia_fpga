//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

#define PREPARE_UNROLL 8
#define REDUCTION_UNROLL 6
#define FADD_LATENCY 8
#define REDUCE_LATENCY (REDUCTION_UNROLL/2)*FADD_LATENCY
#define SRAD_UNROLL 2
#define SRAD2_UNROLL 2

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"
#include "../common/opencl_kernel_common.h"

//======================================================================================================================================================150
//	End
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	Extract KERNEL
//========================================================================================================================================================================================================200

__kernel void extract_kernel(long         d_Ne,
                    __global fp* RESTRICT d_I)				// pointer to input image (DEVICE GLOBAL MEMORY)
{
	for (long ei = 0; ei < d_Ne; ++ei)
	{
		d_I[ei] = exp(d_I[ei]/255);					// exponentiate input IMAGE and copy to output image
	}
}

//========================================================================================================================================================================================================200
//	Prepare KERNEL
//========================================================================================================================================================================================================200

__kernel void prepare_kernel(long         d_Ne,
                    __global fp* RESTRICT d_I,
                    __global fp* RESTRICT d_sums,
                    __global fp* RESTRICT d_sums2)
{
	#pragma unroll PREPARE_UNROLL
	for (long ei = 0; ei < d_Ne; ++ei)
	{        
		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei]*d_I[ei];
	}

}

//========================================================================================================================================================================================================200
//	Reduce KERNEL
//========================================================================================================================================================================================================200

__kernel void reduce_kernel(long         d_Ne,					// number of elements in 
                   __global fp* RESTRICT d_sums,				// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
                   __global fp* RESTRICT d_sums2)
{
	long i;
	int j;
	fp sum1 = 0, sum2 = 0;
	fp shift_reg1[REDUCE_LATENCY+1], shift_reg2[REDUCE_LATENCY+1];

	#pragma unroll
	for (j = 0; j < REDUCE_LATENCY+1; j++)
	{
		shift_reg1[j] = 0;
		shift_reg2[j] = 0;
	}
  
	#pragma unroll REDUCTION_UNROLL
	for (i = 0; i < d_Ne; ++i)
	{
		shift_reg1[REDUCE_LATENCY] = shift_reg1[0] + d_sums[i];
		shift_reg2[REDUCE_LATENCY] = shift_reg2[0] + d_sums2[i];
		
		#pragma unroll
		for (j = 0; j < REDUCE_LATENCY; j++)
		{
			shift_reg1[j] = shift_reg1[j+1];
			shift_reg2[j] = shift_reg2[j+1];
		}
	}
  
	#pragma unroll
	for(j = 0; j < REDUCE_LATENCY; j++)
	{
		sum1 += shift_reg1[j];
		sum2 += shift_reg2[j];
	}
	
	d_sums[0]  = sum1;
	d_sums2[0] = sum2;
}

//========================================================================================================================================================================================================200
//	SRAD KERNEL
//========================================================================================================================================================================================================200

// BUG, IF STILL PRESENT, COULD BE SOMEWHERE IN THIS CODE, MEMORY ACCESS OUT OF BOUNDS
__kernel void srad_kernel(int           d_Nr, 
                          int           d_Nc, 
                 __global int* RESTRICT d_iN, 
                 __global int* RESTRICT d_iS, 
                 __global int* RESTRICT d_jE, 
                 __global int* RESTRICT d_jW, 
                 __global fp*  RESTRICT d_dN, 
                 __global fp*  RESTRICT d_dS, 
                 __global fp*  RESTRICT d_dE, 
                 __global fp*  RESTRICT d_dW, 
                          fp            d_q0sqr, 
                 __global fp*  RESTRICT d_c, 
                 __global fp*  RESTRICT d_I)
{
	for (int col = 0; col < d_Nc; ++col)
	{
		#pragma unroll SRAD_UNROLL
		for (int row = 0; row < d_Nr; ++row)
		{
			int ei = col * d_Nr + row;

			fp d_Jc;
			fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
			fp d_c_loc;
			fp d_G2,d_L,d_num,d_den,d_qsqr;

			// directional derivatives, ICOV, diffusion coefficent
			d_Jc = d_I[ei];								// get value of the current element

			// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
			d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;				// north direction derivative
			d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;				// south direction derivative
			d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;				// west direction derivative
			d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;				// east direction derivative

			// normalized discrete gradient mag squared (equ 52,53)
			d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);

			// normalized discrete laplacian (equ 54)
			d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;		// laplacian (based on derivatives)

			// ICOV (equ 31/35)
			d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;				// num (based on gradient and laplacian)
			d_den  = 1 + (0.25*d_L);							// den (based on laplacian)
			d_qsqr = d_num/(d_den*d_den);						// qsqr (based on num and den)

			// diffusion coefficent (equ 33) (every element of IMAGE)
			d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;			// den (based on qsqr and q0sqr)
			d_c_loc = 1.0 / (1.0+d_den) ;						// diffusion coefficient (based on den)
 
			// saturate diffusion coefficent to 0-1 range
			if (d_c_loc < 0)
			{
				d_c_loc = 0;
			}
			else if (d_c_loc > 1)
			{
				d_c_loc = 1;
			}

			// save data to global memory
			d_dN[ei] = d_dN_loc; 
			d_dS[ei] = d_dS_loc; 
			d_dW[ei] = d_dW_loc; 
			d_dE[ei] = d_dE_loc;
			d_c[ei] = d_c_loc;
		}
	}
}

//========================================================================================================================================================================================================200
//	SRAD2 KERNEL
//========================================================================================================================================================================================================200

// BUG, IF STILL PRESENT, COULD BE SOMEWHERE IN THIS CODE, MEMORY ACCESS OUT OF BOUNDS

__kernel void srad2_kernel(fp            d_lambda, 
                           int           d_Nr, 
                           int           d_Nc, 
                  __global int* RESTRICT d_iS, 
                  __global int* RESTRICT d_jE, 
                  __global fp*  RESTRICT d_dN, 
                  __global fp*  RESTRICT d_dS, 
                  __global fp*  RESTRICT d_dE, 
                  __global fp*  RESTRICT d_dW, 
                  __global fp*  RESTRICT d_c, 
                  __global fp*  RESTRICT d_I,
                  __global fp*  RESTRICT d_I_out)
{
	for (int col = 0; col < d_Nc; ++col)
	{
		#pragma unroll SRAD2_UNROLL
		for (int row = 0; row < d_Nr; ++row)
		{
			int ei = col * d_Nr + row;
			
			fp d_cN,d_cS,d_cW,d_cE;
			fp d_D;

			// diffusion coefficent
			d_cN = d_c[ei];						// north diffusion coefficient
			d_cS = d_c[d_iS[row] + d_Nr*col];			// south diffusion coefficient
			d_cW = d_c[ei];						// west diffusion coefficient
			d_cE = d_c[row + d_Nr * d_jE[col]];			// east diffusion coefficient

			// divergence (equ 58)
			d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];

			// image update (equ 61) (every element of IMAGE)
			d_I_out[ei] = d_I[ei] + 0.25*d_lambda*d_D;
			// updates image (based on input time step and divergence)
		}
	}
}

//========================================================================================================================================================================================================200
//	Compress KERNEL
//========================================================================================================================================================================================================200

__kernel void compress_kernel(long         d_Ne,
                     __global fp* RESTRICT d_I)
{
	for (long i = 0; i < d_Ne; ++i)
	{
		// copy input to output & log uncompress
		d_I[i] = log(d_I[i])*255;					// exponentiate input IMAGE and copy to output image
	}
}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
