#include "../common/common.h"

#define FADD_LATENCY 8

#ifndef DIA_UNROLL
	#define DIA_UNROLL 1
#endif

#ifndef PERI_UNROLL
	#define PERI_UNROLL 1
#endif

#ifndef INT_UNROLL
	#define INT_UNROLL 2
#endif

#ifndef MEM_UNROLL
	#define MEM_UNROLL 16
#endif

#define AA(i,j) matrix[matrix_dim * (i + offset) + j + offset]
#define BB(i,j) matrix[i * matrix_dim + j]


__attribute__((max_global_work_dim(0)))
__kernel void lud_diagonal(__global float* restrict matrix, 
                                    int             matrix_dim, 
                                    int             offset)
{
	for (int i = 0; i < BSIZE; i++)
	{
		int exit = (i % DIA_UNROLL == 0) ? i/DIA_UNROLL : i/DIA_UNROLL + 1;

		// ivdep is added to avoid false dependency on the AA access
		// k is always smaller than both i and j; hence, AA(i, index) or AA(index, j) will never become equal to AA(i, j)
		#pragma ivdep array(matrix)
		for (int j = i; j < BSIZE; j++)
		{
			float shift_reg[FADD_LATENCY] = {0.0f};

			for (int k = 0; k < exit; k++)
			{
				float sum = 0.0f;

				#pragma unroll
				for (int m = 0; m < DIA_UNROLL; m++)
				{
					int index = k * DIA_UNROLL + m;
					if (index < i)
					{
						sum += AA(i, index) * AA(index, j);
					}
				}
				shift_reg[FADD_LATENCY - 1] = shift_reg[0] - sum;

				// shifting
				#pragma unroll
				for (int l = 0; l < FADD_LATENCY - 1; l++)
				{
					shift_reg[l] = shift_reg[l + 1];
				}
			}

			//final reduction
			#pragma unroll
			for (int l = 0; l < FADD_LATENCY - 1; l++)
			{
				AA(i, j) += shift_reg[l];
			}
		}
   
		float temp = 1.0f / AA(i,i);
		// ivdep is added to avoid false dependency on the AA access
		// k is always smaller than both i and j; hence, AA(i, index) or AA(index, j) will never become equal to AA(i, j)
		#pragma ivdep array(matrix)
		for (int j = i + 1; j < BSIZE; j++)
		{
			float shift_reg[FADD_LATENCY] = {0.0f};

			for (int k = 0; k < exit; k++)
			{
				float sum = 0.0f;

				#pragma unroll
				for (int m = 0; m < DIA_UNROLL; m++)
				{
					int index = k * DIA_UNROLL + m;
					if (index < i)
					{
						sum += AA(j, index) * AA(index, i);
					}
				}
				shift_reg[FADD_LATENCY - 1] = shift_reg[0] - sum;

				// shifting
				#pragma unroll
				for (int l = 0; l < FADD_LATENCY - 1; l++)
				{
					shift_reg[l] = shift_reg[l + 1];
				}
			}

			//final reduction
			#pragma unroll
			for (int l = 0; l < FADD_LATENCY - 1; l++)
			{
				AA(j, i) += shift_reg[l];
			}
			AA(j, i) = AA(j, i) * temp;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void lud_perimeter(__global float* restrict matrix, 
                                     int             matrix_dim, 
                                     int             offset,
                                     int             offset_2)
{
	float temp[BSIZE * BSIZE];

	for (int i = 0; i < BSIZE; i++)
	{
		#pragma unroll MEM_UNROLL
		for (int j = 0; j < BSIZE; j++)
		{
			temp[i * BSIZE + j] = AA(i, j);
		}
	}

	// processing top perimeter
	for (int j = 0; j < BSIZE; j++)
	{
		for (int i = 0; i < BSIZE; i++)
		{
			float shift_reg[FADD_LATENCY] = {0.0f};

			int exit = (i % PERI_UNROLL == 0) ? i/PERI_UNROLL : i/PERI_UNROLL + 1;

			for (int k = 0; k < exit; k++)
			{
				float sum = 0.0f;

				#pragma unroll
				for (int m = 0; m < PERI_UNROLL; m++)
				{
					int index = k * PERI_UNROLL + m;
					if (index < i)
					{
						sum += temp[BSIZE * i + index] * BB((offset + index), (offset_2 + j));
					}
				}

				// reduction
				shift_reg[FADD_LATENCY - 1] = shift_reg[0] + sum;

				// shifting
				#pragma unroll
				for (int l = 0; l < FADD_LATENCY - 1; l++)
				{
					shift_reg[l] = shift_reg[l + 1];
				}
			}

			// final reduction
			float sum_final = 0.0f;
			#pragma unroll
			for (int l = 0; l < FADD_LATENCY - 1; l++)
			{
				sum_final += shift_reg[l];
			}

			int i_here = offset + i;
			int j_here = offset_2 + j;
			BB(i_here, j_here) = BB(i_here, j_here) - sum_final;
		}
	}

	// processing left perimeter
	for (int i = 0; i < BSIZE; i++)
	{
		for (int j = 0; j < BSIZE; j++)
		{
			float shift_reg[FADD_LATENCY] = {0.0f};

			int exit = (j % PERI_UNROLL == 0) ? j/PERI_UNROLL : j/PERI_UNROLL + 1;

			for (int k = 0; k < exit; k++)
			{
				float sum = 0.0f;

				#pragma unroll
				for (int m = 0; m < PERI_UNROLL; m++)
				{
					int index = k * PERI_UNROLL + m;
					if (index < j)
					{
						sum += BB((offset_2 + i), (offset + index)) * temp[BSIZE * index + j];
					}
				}

				// reduction
				shift_reg[FADD_LATENCY - 1] = shift_reg[0] + sum;

				// shifting
				#pragma unroll
				for (int l = 0; l < FADD_LATENCY - 1; l++)
				{
					shift_reg[l] = shift_reg[l + 1];
				}
			}

			// final reduction
			float sum_final = 0.0f;
			#pragma unroll
			for (int l = 0; l < FADD_LATENCY - 1; l++)
			{
				sum_final += shift_reg[l];
			}

			int i_here = offset_2 + i;
			int j_here = offset + j;
			BB(i_here, j_here) = (BB(i_here, j_here) - sum_final) / AA(j, j);
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void lud_internal(__global float* restrict matrix, 
                                    int             matrix_dim,
                                    int             offset,
                                    int             i_global,
                                    int             j_global)
{
	float temp_top[BSIZE * BSIZE];
	float temp_left[BSIZE * BSIZE];
	  
	for (int i = 0; i < BSIZE; i++)
	{
		#pragma unroll MEM_UNROLL
		for (int j = 0; j < BSIZE; j++)
		{
			temp_top[i * BSIZE + j]  = matrix[matrix_dim * (i + offset) + j + j_global];
			temp_left[i * BSIZE + j] = matrix[matrix_dim * (i + i_global) + offset + j];
		}
	}

	// use ivdep since the BB address never repeats
	#pragma ivdep array(matrix)
	for (int i = 0; i < BSIZE; i++)
	{
		#pragma unroll INT_UNROLL
		for (int j = 0; j < BSIZE; j++)
		{
			float sum = 0.0f;
			#pragma unroll
			for (int k = 0; k < BSIZE; k++)
			{
				sum += temp_left[BSIZE * i + k] * temp_top[BSIZE * k + j];
			}
			BB((i + i_global), (j + j_global)) -= sum;
		}
	}
}
