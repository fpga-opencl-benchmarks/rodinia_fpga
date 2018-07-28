#ifndef BSIZE
	#define BSIZE 32
#endif

#define FADD_LATENCY 8

#ifndef DIA_UNROLL
	#define DIA_UNROLL 1
#endif
#define DIA_REDUCTION_LATENCY ((DIA_UNROLL + (DIA_UNROLL % 2)) / 2) * FADD_LATENCY

#ifndef PER_UNROLL
	#define PER_UNROLL 1
#endif
#define PER_REDUCTION_LATENCY ((PER_UNROLL + (PER_UNROLL % 2)) / 2) * FADD_LATENCY

#ifndef INT_UNROLL
	#define INT_UNROLL 1
#endif

#define AA(i,j) m[offset * matrix_dim + i * matrix_dim + j + offset]
#define BB(i,j) m[i * matrix_dim + j]

#include "../common/opencl_kernel_common.h"

__attribute__((max_global_work_dim(0)))
__kernel void lud_diagonal(__global float* RESTRICT m, 
                                    int             matrix_dim, 
                                    int             offset)
{ 
	int i, j, k, l;
	float shift_reg[DIA_REDUCTION_LATENCY + 1];

	for (i = 0; i < BSIZE; i++)
	{
		for (j = i; j < BSIZE; j++)
		{
			// initiliaze shift register
			#pragma unroll
			for (l = 0; l < DIA_REDUCTION_LATENCY + 1; l++)
			{
				shift_reg[l] = 0;
			}

			#pragma unroll DIA_UNROLL
			for (k = 0; k < i; k++)
			{
				shift_reg[DIA_REDUCTION_LATENCY] = shift_reg[0] - AA(i, k) * AA(k, j);

				// shifting
				#pragma unroll
				for (l = 0; l < DIA_REDUCTION_LATENCY; l++)
				{
					shift_reg[l] = shift_reg[l + 1];
				}
			}

			//final reduction
			#pragma unroll
			for (l = 0; l < DIA_REDUCTION_LATENCY; l++)
			{
				AA(i, j) += shift_reg[l];
			}
		}
   
		float temp = 1.0f / AA(i,i);
		for (j = i + 1; j < BSIZE; j++)
		{
			// initiliaze shift register
			#pragma unroll
			for (l = 0; l < DIA_REDUCTION_LATENCY + 1; l++)
			{
				shift_reg[l] = 0;
			}

			#pragma unroll DIA_UNROLL
			for (k = 0; k < i; k++)
			{
				shift_reg[DIA_REDUCTION_LATENCY] = shift_reg[0] - AA(j, k) * AA(k, i);

				// shifting
				#pragma unroll
				for (l = 0; l < DIA_REDUCTION_LATENCY; l++)
				{
					shift_reg[l] = shift_reg[l + 1];
				}
			}

			//final reduction
			#pragma unroll
			for (l = 0; l < DIA_REDUCTION_LATENCY; l++)
			{
				AA(j, i) += shift_reg[l];
			}
			AA(j, i) = AA(j, i) * temp;
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void lud_perimeter(__global float* RESTRICT m, 
                                     int             matrix_dim, 
                                     int             offset)
{
	int chunk_idx, chunk_num;
	float shift_reg[PER_REDUCTION_LATENCY + 1];

	chunk_num = ((matrix_dim - offset) / BSIZE) - 1;

	for (chunk_idx = 0; chunk_idx < chunk_num; chunk_idx++)
	{
		int i, j, k, l, i_global, j_global, i_here, j_here;
		float sum;           
		float temp[BSIZE * BSIZE];

		for (i = 0; i < BSIZE; i++)
		{
			for (j = 0; j < BSIZE; j++)
			{
				temp[i * BSIZE + j] = AA(i, j);
			}
		}

		i_global = offset;
		j_global = offset;
            
		// processing top perimeter
		j_global += BSIZE * (chunk_idx + 1);
		for (j = 0; j < BSIZE; j++)
		{
			for (i = 0; i < BSIZE; i++)
			{
				// initiliaze shift register
				#pragma unroll
				for (l = 0; l < PER_REDUCTION_LATENCY + 1; l++)
				{
					shift_reg[l] = 0;
				}

				#pragma unroll PER_UNROLL
				for (k = 0; k < i; k++)
				{
					// reduction
					shift_reg[PER_REDUCTION_LATENCY] = shift_reg[0] + temp[BSIZE * i + k] * BB((i_global + k), (j_global + j));

					// shifting
					#pragma unroll
					for (l = 0; l < PER_REDUCTION_LATENCY; l++)
					{
						shift_reg[l] = shift_reg[l + 1];
					}
				}

				// final reduction
				sum = 0.0f;
				#pragma unroll
				for (l = 0; l < PER_REDUCTION_LATENCY; l++)
				{
					sum += shift_reg[l];
				}

				i_here = i_global + i;
				j_here = j_global + j;
				BB(i_here, j_here) = BB(i_here, j_here) - sum;
			}
		}

		// processing left perimeter
		j_global = offset;
		i_global += BSIZE * (chunk_idx + 1);
		for (i = 0; i < BSIZE; i++)
		{
			for (j = 0; j < BSIZE; j++)
			{
				// initiliaze shift register
				#pragma unroll
				for (l = 0; l < PER_REDUCTION_LATENCY + 1; l++)
				{
					shift_reg[l] = 0;
				}

				#pragma unroll PER_UNROLL
				for (k = 0; k < j; k++)
				{
					// reduction
					shift_reg[PER_REDUCTION_LATENCY] = shift_reg[0] + BB((i_global + i), (j_global + k)) * temp[BSIZE * k + j];

					// shifting
					#pragma unroll
					for (l = 0; l < PER_REDUCTION_LATENCY; l++)
					{
						shift_reg[l] = shift_reg[l + 1];
					}
				}

				// final reduction
				sum = 0.0f;
				#pragma unroll
				for (l = 0; l < PER_REDUCTION_LATENCY; l++)
				{
					sum += shift_reg[l];
				}

				i_here = i_global + i;
				j_here = j_global + j;
				BB(i_here, j_here) = (BB(i_here, j_here) - sum) / AA(j, j);
			}
		}
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void lud_internal(__global float* RESTRICT m, 
                                    int             matrix_dim, 
                                    int             offset)
{
	int chunk_idx, chunk_num;

	chunk_num = ((matrix_dim - offset) / BSIZE) - 1;

	for  (chunk_idx = 0; chunk_idx < chunk_num * chunk_num; chunk_idx++)
	{
		int i, j, k, i_global, j_global;
		float temp_top[BSIZE * BSIZE];
		float temp_left[BSIZE * BSIZE];
		float sum[BSIZE] = {0.0f};
            
		i_global = offset + BSIZE * (1 + chunk_idx / chunk_num);
		j_global = offset + BSIZE * (1 + chunk_idx % chunk_num);

		for (i = 0; i < BSIZE; i++)
		{
			for (j = 0; j < BSIZE; j++)
			{
				temp_top[i * BSIZE + j]  = m[matrix_dim * (i + offset) + j + j_global];
				temp_left[i * BSIZE + j] = m[matrix_dim * (i + i_global) + offset + j];
			}
		}

		for (i = 0; i < BSIZE; i++)
		{
			for (j = 0; j < BSIZE; j++)
			{
				sum[j] = 0.0f;
				#pragma unroll
				for (k = 0; k < BSIZE; k++)
				{
					sum[j] += temp_left[BSIZE * i + k] * temp_top[BSIZE * k + j];
				}
			}

			#pragma unroll
			for (j = 0; j < BSIZE; j++)
			{
				BB((i + i_global), (j + j_global)) -= sum[j];
			}
		}
	}
}
