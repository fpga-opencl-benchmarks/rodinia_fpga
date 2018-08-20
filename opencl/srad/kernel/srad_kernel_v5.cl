//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

#define FADD_LATENCY	24 // deliberately increased to allow correct pipelining for high target fmax values

#define C_LOC_BASE_SIZE  BSIZE
#define C_LOC_SR_SIZE    C_LOC_BASE_SIZE + SSIZE
#define I_BASE_SIZE      2 * BSIZE
#define I_SR_SIZE        I_BASE_SIZE + SSIZE

#define SR_OFFSET        BSIZE
#define SR_OFFSET_N      SR_OFFSET + 1
#define SR_OFFSET_S      SR_OFFSET - 1
#define SR_OFFSET_W      SR_OFFSET + BSIZE
#define SR_OFFSET_E      SR_OFFSET - BSIZE

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"

//======================================================================================================================================================150
//	End
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	Compute KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void compute_kernel(fp           lambda,
                             int          Nr, 
                             int          Nc,
                             long         Ne,
                             int          red_exit,
                             int          comp_exit,
                             int          last_row,
                    __global fp* restrict I,
                    __global fp* restrict I_out)
{
	//======================================================================================================================================================150
	//	Reduction
	//======================================================================================================================================================150

	fp final_sum_1 = 0.0f, final_sum_2 = 0.0f;						// reduction kernel variables

#ifdef AOCL_BOARD_de5net_a7 // shift register to optimize reduction for Stratix V
	fp shift_reg_1[FADD_LATENCY] = {0.0f};
	fp shift_reg_2[FADD_LATENCY] = {0.0f};

	for (int i = 0; i < red_exit; i++)
	{
		fp sum_1 = 0.0f, sum_2 = 0.0f;
		#pragma unroll
		for (int j = 0; j < RED_UNROLL; j++)
		{
			long index = i * RED_UNROLL + j;
			
			sum_1 += (index < Ne) ? I[index] : 0.0;
			sum_2 += (index < Ne) ? I[index] * I[index] : 0.0;
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
	#pragma unroll
	for (int i = 0; i < FADD_LATENCY - 1; i++)
	{
		final_sum_1 += shift_reg_1[i];
		final_sum_2 += shift_reg_2[i];
	}
#else // single cycle floating-point accumulation for Arria 10
	for (int i = 0; i < red_exit; i++)
	{
		fp sum_1 = 0.0f, sum_2 = 0.0f;
		#pragma unroll
		for (int j = 0; j < RED_UNROLL; j++)
		{
			long index = i * RED_UNROLL + j;

			sum_1 += (index < Ne) ? I[index] : 0.0;
			sum_2 += (index < Ne) ? I[index] * I[index] : 0.0;
		}

		final_sum_1 += sum_1;
		final_sum_2 += sum_2;
	}
#endif

	fp mean  = final_sum_1 / Ne;									// mean (average) value of element
	fp var   = (final_sum_2 / Ne) - mean * mean;						// variance
	fp q0sqr = var / (mean * mean);								// standard deviation

	//======================================================================================================================================================150
	//	SRAD Compute
	//======================================================================================================================================================150

	fp c_loc_SR[C_LOC_SR_SIZE];									// shift register for c_loc to store and reuse east and south neighbors
	fp I_SR[I_SR_SIZE];											// shift register to store and reuse values of I

	// initialize shift registers
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

	// starting points
	int col = Nc;												// start from an out-of-bound column to fill shift register before valid compute starts
	int block_row_offset = 0;									// starting point of each unrolled chunk of rows in block, bottom row in block is zero
	int block_offset = last_row - 1;								// starting row of the block
	int index = 0;												// global index

	while (index != comp_exit)
	{
		index++;												// increment global index

		// shift both shift registers by SSIZE
		#pragma unroll
		for (int i = 0; i < I_BASE_SIZE; i++)
		{
			I_SR[i] = I_SR[i + SSIZE];
		}

		#pragma unroll
		for (int i = 0; i < C_LOC_BASE_SIZE; i++)
		{
			c_loc_SR[i] = c_loc_SR[i + SSIZE];
		}

		#pragma unroll
		for (int i = 0; i < SSIZE; i++)
		{
			int block_row = block_row_offset + i - HALO;				// row in compute block
			int row = block_offset - block_row;					// global row
			int read_col = col - 1;								// column for reading from memory
			long read_offset = read_col * Nr + row;					// index offset for reading from memory, transposed to match original code
			long comp_offset = read_offset + Nr;					// index for compute and write to memory

			fp dN, dS, dW, dE;									// original SRAD1 kernel temp variables
			fp cC, cN, cS, cW, cE;								// original SRAD2 kernel temp variables

			if (row >= 0 && row < Nr && col != 0)					// avoid out-of-bound indexes in y direction and also nothing to read on the leftmost column
			{
				I_SR[I_BASE_SIZE + i] = I[read_offset];				// read new values from memory to shift register
			}

			// directional derivatives, ICOV, diffusion coefficient
			cC = I_SR[SR_OFFSET + i];							// current index

			// directional derivatives
			// north direction derivative
			if (row == 0)										// top image border
			{
				dN = 0;										// I[comp_offset] - I[comp_offset]
			}
			else
			{
				dN = I_SR[SR_OFFSET_N + i] - cC;					// read from shift register
			}

			// south direction derivative
			if (row == Nr - 1)									// bottom image border
			{
				dS = 0;										// I[comp_offset] - I[comp_offset]
			}
			else
			{
				dS = I_SR[SR_OFFSET_S + i] - cC;					// read from shift register
			}

			// west direction derivative
			if (col == 0)										// left image border
			{
				dW = 0;										// I[comp_offset] - I[comp_offset]
			}
			else
			{
				dW = I_SR[SR_OFFSET_W + i] - cC;					// read from shift register
			}

			// east direction derivative
			if (col == Nc - 1)									// right image border
			{
				dE = 0;										// I[comp_offset] - I[comp_offset]
			}
			else
			{
				dE = I_SR[SR_OFFSET_E + i] - cC;					// read from shift register
			}

			// normalized discrete gradient mag squared (equ 52,53)
			fp G2 = (dN * dN + dS * dS + dW * dW + dE * dE) / (cC * cC);

			// normalized discrete laplacian (equ 54)
			fp L = (dN + dS + dW + dE) / cC;						// laplacian (based on derivatives)

			// ICOV (equ 31/35)
			fp num  = (G2 / 2.0) - ((L * L) / 16.0);				// num (based on gradient and laplacian)
			fp denl = 1 + (L / 4.0);								// den (based on laplacian)
			fp qsqr = num / (denl * denl);						// qsqr (based on num and den)

			// diffusion coefficient (equ 33)
			fp denq = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));			// den (based on qsqr and q0sqr)
			fp c_loc = 1.0 / (1.0 + denq);						// diffusion coefficient (based on den)

			// clamp diffusion coefficient if out of 0-1 range
			if (c_loc < 0)
			{
				c_loc = 0;
			}
			else if (c_loc > 1)
			{
				c_loc = 1;
			}

			// the following write does not need any conditions to avoid halos or out-of-bound cells since halo size 
			// has been increased to account for the two stencils in the kernel.
			// the invalid halo computation from the first stencil will not propagate to the final output since the halo of the
			// second stencil is larger than the first one
			c_loc_SR[C_LOC_BASE_SIZE + i] = cN = cW = c_loc;			// write new value to shift register to be reused later

			// diffusion coefficient
			// south diffusion coefficient
			if (row == Nr - 1)									// bottom image boundary
			{
				cS = c_loc;
			}
			else
			{
				cS = c_loc_SR[SR_OFFSET_S + i];					// read from shift register
			}

			// east diffusion coefficient
			if (col == Nc - 1)									// right image boundary
			{
				cE = c_loc;
			}
			else
			{
				cE = c_loc_SR[SR_OFFSET_E + i];					// read from shift register
			}

			// divergence (equ 58)
			fp D = cN * dN + cS * dS + cW * dW + cE * dE;

			// write output values to memory
			// the following condition is to avoid halos and going out of bounds in all axes
			if (col != Nc && row < Nr && block_row >= 0 && block_row < BSIZE - 2 * HALO)
			{
				// updates image based on input time step and divergence (equ 61)
				I_out[comp_offset] = cC + (lambda * D) / 4.0;
			}
		}

		// "& (BSIZE - 1)" replaces "% BSIZE" since BSIZE is a power of two
		block_row_offset = (block_row_offset + SSIZE) & (BSIZE - 1);	// move one chunk forward and reset to zero if end of block column was reached

		if (block_row_offset == 0)								// if one column finished
		{
			if (col == 0)										// if on last column (compute traverses one more column than memory read/write)
			{
				col = Nc;										// reset column back to original value
				block_offset -= BSIZE - 2 * HALO;					// go to next block, account for overlapping
			}
			else
			{
				col--;										// go to next column
			}
		}
	}
}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
