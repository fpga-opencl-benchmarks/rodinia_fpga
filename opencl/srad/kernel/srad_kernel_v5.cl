//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

#define REDUCTION_UNROLL 16
#define FADLATENCY 8
#define REDUCE_LATENCY (REDUCTION_UNROLL/2)*FADLATENCY
#ifndef BSIZE
	#if defined(AOCL_BOARde5net_a7)
		#define BSIZE 128
	#endif
#endif
#ifndef SSIZE
	#if defined(AOCL_BOARde5net_a7)
		#define SSIZE 4
	#endif
#endif

#define C_LOC_BASE_SIZE  BSIZE
#define C_LOC_SR_SIZE    C_LOC_BASE_SIZE + SSIZE
#define I_BASE_SIZE      2 * BSIZE
#define I_SR_SIZE        I_BASE_SIZE + SSIZE

#define sw_offset        BSIZE
#define sw_offset_n      sw_offset + 1
#define sw_offset_s      sw_offset - 1
#define sw_offset_w      sw_offset + BSIZE
#define sw_offset_e      sw_offset - BSIZE

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

__attribute__((max_global_work_dim(0)))
__kernel void extract_kernel(long         Ne,
                    __global fp* RESTRICT I)							// pointer to input image (DEVICE GLOBAL MEMORY)
{
	for (long ei = 0; ei < Ne; ++ei)
	{
		I[ei] = exp(I[ei]/255);								// exponentiate input IMAGE and copy to output image
	}
}

//========================================================================================================================================================================================================200
//	Compute KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void compute_kernel(fp            lambda,
                             int           Nr, 
                             int           Nc,
           __global volatile fp*  RESTRICT I,							// marked as volatile to disable Altera's cache; the cache only wastes area since we are caching this value in I_SR anyway
           __global          fp*  RESTRICT c_boundary,
           __global          fp*  RESTRICT I_out)
{
	fp sums = 0, sums2 = 0;									// reduction kernel variables
	fp shift_reg1[REDUCE_LATENCY + 1], shift_reg2[REDUCE_LATENCY + 1];			// shift register to optimize reduction for FPGA
	fp c_loc_SR[C_LOC_SR_SIZE];								// shift register for c_loc to resolve right and bottom dependency locally
	fp I_SR[I_SR_SIZE];									// shift register for storing values of I to reduce global memory access

	long Ne = Nr * Nc;

	//======================================================================================================================================================150
	//	Reduction
	//======================================================================================================================================================150
	#pragma unroll
	for (int i = 0; i < REDUCE_LATENCY + 1; i++)
	{
		shift_reg1[i] = 0;
		shift_reg2[i] = 0;
	}
  
	#pragma unroll REDUCTION_UNROLL
	for (long i = 0; i < Ne; ++i)
	{
		shift_reg1[REDUCE_LATENCY] = shift_reg1[0] + I[i];
		shift_reg2[REDUCE_LATENCY] = shift_reg2[0] + I[i] * I[i];
		
		#pragma unroll
		for (int j = 0; j < REDUCE_LATENCY; j++)
		{
			shift_reg1[j] = shift_reg1[j + 1];
			shift_reg2[j] = shift_reg2[j + 1];
		}
	}
  
	#pragma unroll
	for (int j = 0; j < REDUCE_LATENCY; j++)
	{
		sums  += shift_reg1[j];
		sums2 += shift_reg2[j];
	}

	fp mean   = sums / Ne;									// mean (average) value of element
	fp var    = (sums2 / Ne) - mean * mean;							// variance
	fp q0sqr  = var / (mean * mean);							// standard deviation

	//======================================================================================================================================================150
	//	SRAD Compute
	//======================================================================================================================================================150

	// initializing shift registers
	#pragma unroll
	for (int i = 0; i < I_SR_SIZE; i++)
	{
		I_SR[i] = 0;
	}

	#pragma unroll
	for (int i = 0; i < C_LOC_SR_SIZE; i++)
	{
		c_loc_SR[i] = 0;
	}
	
	int block_num = (Nr % BSIZE == 0) ? Nr / BSIZE : (Nr / BSIZE) + 1;				// number of blocks is calculated by dividing column size by BSIZE and adding one if not divisible

	// the below loop on blocks can be merged into the while loop, but it shouldn't
	// the only way to guarantee that c_boundary will be written in one block,
	// before it is read in the next, is for blocks to be processed sequentially
	// this only happens if the loop on blocks is outside of the while loop
	for (int i = block_num - 1; i >= 0; --i)
	{
		int col = Nc;										// col = Nc is only for loading data, no computation is done
		int row_offset = BSIZE - 1;								// start from bottom row in the block
		
		#pragma ivdep array(c_boundary)								// to fix false dependency on the c_boundary array
		do
		{
			fp boundary;

			// shift both shift registers by SSIZE
			#pragma unroll
			for (int j = 0; j < I_BASE_SIZE; j++)
			{
				I_SR[j] = I_SR[j + SSIZE];
			}

			#pragma unroll
			for (int j = 0; j < C_LOC_BASE_SIZE; j++)
			{
				c_loc_SR[j] = c_loc_SR[j + SSIZE];
			}

			#pragma unroll
			for (int j = 0; j < SSIZE; j++)
			{
				uint block_row = row_offset - j;					// row number in the current block
				uint row = i * BSIZE + block_row;					// real global row number

				if (row < Nr && col != 0)						// no need to read anything from memory on the border or when we are out-of-bound
				{
					ulong readoffset = (col - 1) * Nr + row;			// offset for reading from memory
					I_SR[I_BASE_SIZE + j] = I[readoffset];				// read new values from memory, reads column i-1 (west dependency) when computing column i
				}

				if (row < Nr && col != Nc)						// avoid going out-of-bounds in either axis
				{
					fp dN, dS, dW, dE;						// original SRAD1 kernel temp variables
					fp cC, cN, cS, cW, cE;						// original SRAD2 kernel temp variables
					ulong ei = col * Nr + row;

					// directional derivatives, ICOV, diffusion coefficient
					cC = I_SR[sw_offset + j];					// current index

					// directional derivatives
					// north direction derivative
					if (row == 0)							// top image border
					{
						dN = 0;							// I[ei] - I[ei]
					}
					else if (row_offset == SSIZE - 1 && j == SSIZE - 1)		// top block boundary, avoid using block_row == 0 to prevent extra off-chip memory read ports
					{
						dN = I[ei - 1] - cC;					// read from off-chip memory
					}
					else
					{
						dN = I_SR[sw_offset_n + j] - cC;			// read from shift register
					}

					// south direction derivative
					if (row == Nr - 1)						// bottom image border
					{
						dS = 0;							// I[ei] - I[ei]
					}
					else if (row_offset == BSIZE - 1 && j == 0)			// bottom block boundary, avoid using block_row == BSIZE - 1 to prevent extra off-chip memory read ports
					{
						dS = I[ei + 1] - cC;					// read from off-chip memory
					}
					else
					{
						dS = I_SR[sw_offset_s + j] - cC;			// read from shift register
					}

					// west direction derivative
					if (col == 0)							// left image border
					{
						dW = 0;							// I[ei] - I[ei]
					}
					else
					{
						dW = I_SR[sw_offset_w + j] - cC;			// read from shift register
					}

					// east direction derivative
					if (col == Nc - 1)						// right image border
					{
						dE = 0;							// I[ei] - I[ei]
					}
					else
					{
						dE = I_SR[sw_offset_e + j] - cC;			// read from shift register
					}

					// normalized discrete gradient mag squared (equ 52,53)
					fp G2 = (dN * dN + dS * dS + dW * dW + dE * dE) / (cC * cC);

					// normalized discrete laplacian (equ 54)
					fp L = (dN + dS + dW + dE) / cC;				// laplacian (based on derivatives)

					// ICOV (equ 31/35)
					fp num  = (0.5 * G2) - ((1.0 / 16.0) * (L * L));		// num (based on gradient and laplacian)
					fp denl = 1 + (0.25 * L);					// den (based on laplacian)
					fp qsqr = num / (denl * denl);					// qsqr (based on num and den)

					// diffusion coefficient (equ 33)
					fp denq = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));		// den (based on qsqr and q0sqr)
					fp c_loc = 1.0 / (1.0 + denq);					// diffusion coefficient (based on den)

					// clamp diffusion coefficient if out of 0-1 range
					if (c_loc < 0)
					{
						c_loc = 0;
					}
					else if (c_loc > 1)
					{
						c_loc = 1;
					}

					c_loc_SR[C_LOC_BASE_SIZE + j] = c_loc;				// write new value to shift register to be reused later

					// diffusion coefficient
					// south diffusion coefficient
					if (row == Nr - 1)						// bottom image boundary
					{
						cS = c_loc;
					}
					else if (row_offset == BSIZE - 1 && j == 0)			// bottom block boundary, avoid using block_row == BSIZE - 1 to prevent extra off-chip memory read ports
					{
						cS = c_boundary[col];					// read from the extra boundary buffer
					}
					else
					{
						cS = c_loc_SR[sw_offset_s + j];				// read from shift register
					}

					// east diffusion coefficient
					if (col == Nc - 1)						// right image boundary
					{
						cE = c_loc;
					}
					else
					{
						cE = c_loc_SR[sw_offset_e + j];				// read from shift register
					}

					// divergence (equ 58)
					fp D = c_loc * dN + cS * dS + c_loc * dW + cE * dE;

					// updates image based on input time step and divergence (equ 61)
					I_out[ei] = cC + 0.25 * lambda * D;

					if (row_offset == SSIZE - 1 && j == SSIZE - 1)			// top block boundary, avoid using block_row == 0 to prevent extra read ports
					{
						boundary = c_loc;					// make a backup from this value to be later written to off-chip memory
					}
				}
			}

			row_offset = row_offset - SSIZE;						// go to next chunk
			if (row_offset == -1)								// top of the block
			{
				row_offset = BSIZE - 1;							// reset row_offset
				c_boundary[col] = boundary;						// write backup value to off-chip memory at the top of the block for resolving south dependency in the upper block
				col--;									// move to next column
			}
		} while (col >= 0);
	}
}

//========================================================================================================================================================================================================200
//	Compress KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void compress_kernel(long         Ne,
                     __global fp* RESTRICT I)
{
	for (long i = 0; i < Ne; ++i)
	{
		// copy input to output & log uncompress
		I[i] = log(I[i])*255;								// exponentiate input IMAGE and copy to output image
	}
}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
