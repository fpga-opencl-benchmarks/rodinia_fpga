//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

#define FADD_LATENCY 8

#define PREP_UNROLL 8
#define SRAD_UNROLL 2
#define SRAD2_UNROLL 2

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"

//======================================================================================================================================================150
//	End
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	Prepare KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void prepare_kernel(long          d_Ne,
                    __global fp*  restrict d_I,
                    __global fp*  restrict d_sums,
                    __global fp*  restrict d_sums2)
{
	#pragma unroll PREP_UNROLL
	for (long ei = 0; ei < d_Ne; ++ei)
	{
		d_sums[ei] = d_I[ei];
		d_sums2[ei] = d_I[ei] * d_I[ei];
	}
}

//========================================================================================================================================================================================================200
//	Reduce KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void reduce_kernel(long          d_Ne,					// number of elements in 
                   __global fp*  restrict d_sums,					// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
                   __global fp*  restrict d_sums2,
                            int           exit)
{
	fp shift_reg_1[FADD_LATENCY] = {0.0f};
	fp shift_reg_2[FADD_LATENCY] = {0.0f};

	for (int i = 0; i < exit; i++)
	{
		fp sum_1 = 0.0f, sum_2 = 0.0f;
		#pragma unroll
		for (int j = 0; j < RED_UNROLL; j++)
		{
			long index = i * RED_UNROLL + j;
			
			if (index < d_Ne)
			{
				sum_1 += d_sums[index];
				sum_2 += d_sums2[index];
			}
		}
		shift_reg_1[FADD_LATENCY - 1] = shift_reg_1[0] + sum_1;
		shift_reg_2[FADD_LATENCY - 1] = shift_reg_2[0] + sum_2;

		// shifting
		#pragma unroll
		for (int j = 0; j < FADD_LATENCY - 1; j++)
		{
			shift_reg_1[j] = shift_reg_1[j + 1];
			shift_reg_2[j] = shift_reg_2[j + 1];
		}
	}

	//final reduction
	fp final_sum_1 = 0.0f, final_sum_2 = 0.0f;
	#pragma unroll
	for (int i = 0; i < FADD_LATENCY - 1; i++)
	{
		final_sum_1 += shift_reg_1[i];
		final_sum_2 += shift_reg_2[i];
	}

	d_sums[0]	 = final_sum_1;
	d_sums2[0] = final_sum_2;
}

//========================================================================================================================================================================================================200
//	SRAD KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void srad_kernel(int           d_Nr, 
                          int           d_Nc, 
                 __global int* restrict d_iN, 
                 __global int* restrict d_iS, 
                 __global int* restrict d_jE, 
                 __global int* restrict d_jW, 
                 __global fp*  restrict d_dN, 
                 __global fp*  restrict d_dS, 
                 __global fp*  restrict d_dE, 
                 __global fp*  restrict d_dW, 
                          fp            d_q0sqr, 
                 __global fp*  restrict d_c, 
                 __global fp*  restrict d_I)
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
			fp d_G2, d_L, d_num, d_den, d_qsqr;

			// directional derivatives, ICOV, diffusion coefficient
			d_Jc = d_I[ei];									// get value of the current element

			// directional derivatives (every element of IMAGE)(try to copy to shared memory or temp files)
			d_dN_loc = d_I[d_iN[row] + d_Nr * col] - d_Jc;			// north direction derivative
			d_dS_loc = d_I[d_iS[row] + d_Nr * col] - d_Jc;			// south direction derivative
			d_dW_loc = d_I[row + d_Nr * d_jW[col]] - d_Jc;			// west direction derivative
			d_dE_loc = d_I[row + d_Nr * d_jE[col]] - d_Jc;			// east direction derivative

			// normalized discrete gradient mag squared (equ 52,53)
			d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc + d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) / (d_Jc * d_Jc);

			// normalized discrete laplacian (equ 54)
			d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;	// laplacian (based on derivatives)

			// ICOV (equ 31/35)
			d_num	= (0.5 * d_G2) - ((1.0 / 16.0) * (d_L * d_L));	// num (based on gradient and laplacian)
			d_den	= 1 + (0.25 * d_L);							// den (based on laplacian)
			d_qsqr = d_num / (d_den * d_den);						// qsqr (based on num and den)

			// diffusion coefficient (equ 33) (every element of IMAGE)
			d_den = (d_qsqr - d_q0sqr) / (d_q0sqr * (1 + d_q0sqr));	// den (based on qsqr and q0sqr)
			d_c_loc = 1.0 / (1.0 + d_den) ;						// diffusion coefficient (based on den)
 
			// saturate diffusion coefficient to 0-1 range
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
			d_c[ei]  = d_c_loc;
		}
	}
}

//========================================================================================================================================================================================================200
//	SRAD2 KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void srad2_kernel(fp            d_lambda, 
                           int           d_Nr, 
                           int           d_Nc, 
                  __global int* restrict d_iS, 
                  __global int* restrict d_jE, 
                  __global fp*  restrict d_dN, 
                  __global fp*  restrict d_dS, 
                  __global fp*  restrict d_dE, 
                  __global fp*  restrict d_dW, 
                  __global fp*  restrict d_c, 
                  __global fp*  restrict d_I)
{
	#pragma ivdep array(d_I)
	for (int col = 0; col < d_Nc; ++col)
	{
		#pragma unroll SRAD2_UNROLL
		#pragma ivdep array(d_I)
		for (int row = 0; row < d_Nr; ++row)
		{
			int ei = col * d_Nr + row;
			
			fp d_cN, d_cS, d_cW, d_cE;
			fp d_D;

			// diffusion coefficient
			d_cN = d_c[ei];									// north diffusion coefficient
			d_cS = d_c[d_iS[row] + d_Nr*col];						// south diffusion coefficient
			d_cW = d_c[ei];									// west diffusion coefficient
			d_cE = d_c[row + d_Nr * d_jE[col]];					// east diffusion coefficient

			// divergence (equ 58)
			d_D = d_cN * d_dN[ei] + d_cS * d_dS[ei] + d_cW * d_dW[ei] + d_cE * d_dE[ei];

			// image update (equ 61) (every element of IMAGE)
			d_I[ei] = d_I[ei] + 0.25 * d_lambda * d_D;
			// updates image (based on input time step and divergence)
		}
	}
}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
