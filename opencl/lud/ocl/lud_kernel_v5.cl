#include "../common/common.h"
// peri collapsed exit condition: 64,4 -->4*( (1+(64/4-1)) * (64/4-1)/2) + (4-1) * 64/4

#define FADD_LATENCY 8

#ifndef DIA_UNROLL
	#define DIA_UNROLL 2
#endif

#ifndef PERI_UNROLL
	#define PERI_UNROLL 2
#endif

#ifndef INT_UNROLL
	#define INT_UNROLL 2
#endif

#ifndef MEM_UNROLL
	#define MEM_UNROLL 16
#endif


__attribute__((max_global_work_dim(0)))
__kernel void lud_diagonal(__global float* restrict matrix, 
                                    int             matrix_dim, 
                                    int             offset)
{
	float dia[BSIZE * BSIZE];

	int i1 = 0;
	int j1 = 0;
	int index1 = 0;
	int array_offset = offset * matrix_dim + offset;

	#pragma ivdep
	while (index1 != (BSIZE * BSIZE / MEM_UNROLL))
	{
		#pragma unroll
		for (int j_loc = 0; j_loc < MEM_UNROLL; j_loc++)
		{
			int j_real = j1 * MEM_UNROLL + j_loc;

			dia[i1 * BSIZE + j_real] = matrix[array_offset + j_real];
		}

		if (j1 == (BSIZE / MEM_UNROLL) - 1)
		{
			j1 = 0;
			i1++;
			array_offset += matrix_dim;
		}
		else
		{
			j1++;
		}

		index1++;
	}

	for (int i = 0; i < BSIZE; i++)
	{
		int exit = (i % DIA_UNROLL == 0) ? i/DIA_UNROLL : i/DIA_UNROLL + 1;

		// ivdep is added to avoid false dependency on the dia buffer
		// k is always smaller than both i and j; hence, dia(i, index) or dia(index, j) will never become equal to dia(i, j)
		#pragma ivdep array(dia)
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
						sum += dia[i * BSIZE + index] * dia[index * BSIZE + j];
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
				dia[i * BSIZE + j] += shift_reg[l];
			}
		}

		// ivdep is added to avoid false dependency on the dia buffer
		// k is always smaller than both i and j; hence, dia(i, index) or dia(index, j) will never become equal to dia(i, j)
		#pragma ivdep array(dia)
		for (int j = i + 1; j < BSIZE; j++)
		{
			float shift_reg[FADD_LATENCY] = {0.0f};

			for (int k = 0; k < exit; k++)
			{
				float sum = 0.0f;

				#pragma unroll
				for (int k_loc = 0; k_loc < DIA_UNROLL; k_loc++)
				{
					int k_real = k * DIA_UNROLL + k_loc;
					if (k_real < i)
					{
						sum += dia[j * BSIZE + k_real] * dia[k_real * BSIZE + i];
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

			float sum = dia[j * BSIZE + i];
			//final reduction
			#pragma unroll
			for (int l = 0; l < FADD_LATENCY - 1; l++)
			{
				sum += shift_reg[l];
			}
			dia[j * BSIZE + i] = sum / dia[i * BSIZE + i];
		}
	}

	int i2 = 0;
	int j2 = 0;
	int index2 = 0;
	array_offset = offset * matrix_dim + offset;

	#pragma ivdep
	while (index2 != (BSIZE * BSIZE / MEM_UNROLL))
	{
		#pragma unroll
		for (int j_loc = 0; j_loc < MEM_UNROLL; j_loc++)
		{
			int j_real = j2 * MEM_UNROLL + j_loc;

			matrix[array_offset + j_real] = dia[i2 * BSIZE + j_real];
		}

		if (j2 == (BSIZE / MEM_UNROLL) - 1)
		{
			j2 = 0;
			i2++;
			array_offset += matrix_dim;
		}
		else
		{
			j2++;
		}

		index2++;
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void lud_perimeter(__global float* restrict matrix, 
                                     int             matrix_dim, 
                                     int             offset,
                                     int             offset_2)
{
	float dia[BSIZE * BSIZE], peri_col[BSIZE * BSIZE], peri_row[BSIZE * BSIZE];

	int i1 = 0;
	int j1 = 0;
	int index1 = 0;
	int array_offset_1 = offset * matrix_dim + offset;
	int array_offset_2 = offset * matrix_dim + offset_2;
	int array_offset_3 = offset_2 * matrix_dim + offset;

	#pragma ivdep
	while (index1 != (BSIZE * BSIZE / MEM_UNROLL))
	{
		#pragma unroll
		for (int j_loc = 0; j_loc < MEM_UNROLL; j_loc++)
		{
			int j_real = j1 * MEM_UNROLL + j_loc;

			dia[i1 * BSIZE + j_real] = matrix[array_offset_1 + j_real];
			peri_row[i1 * BSIZE + j_real] = matrix[array_offset_2 + j_real];
			peri_col[i1 * BSIZE + j_real] = matrix[array_offset_3 + j_real];
		}

		if (j1 == (BSIZE / MEM_UNROLL) - 1)
		{
			j1 = 0;
			i1++;
			array_offset_1 += matrix_dim;
			array_offset_2 += matrix_dim;
			array_offset_3 += matrix_dim;
		}
		else
		{
			j1++;
		}

		index1++;
	}

	// processing perimeter row
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
				for (int k_loc = 0; k_loc < PERI_UNROLL; k_loc++)
				{
					int k_real = k * PERI_UNROLL + k_loc;
					if (k_real < i)
					{
						sum += dia[BSIZE * i + k_real] * peri_row[k_real * BSIZE + j];
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

			peri_row[i * BSIZE + j] = peri_row[i * BSIZE + j] - sum_final;
		}
	}

	// processing perimeter column
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
				for (int k_loc = 0; k_loc < PERI_UNROLL; k_loc++)
				{
					int k_real = k * PERI_UNROLL + k_loc;
					if (k_real < j)
					{
						sum += peri_col[i * BSIZE + k_real] * dia[BSIZE * k_real + j];
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
			peri_col[i * BSIZE + j] = (peri_col[i * BSIZE + j] - sum_final) / dia[BSIZE * j + j];
		}
	}

	int i2 = 0;
	int j2 = 0;
	int index2 = 0;
	array_offset_2 = offset * matrix_dim + offset_2;
	array_offset_3 = offset_2 * matrix_dim + offset;

	#pragma ivdep
	while (index2 != (BSIZE * BSIZE / MEM_UNROLL))
	{
		#pragma unroll
		for (int j_loc = 0; j_loc < MEM_UNROLL; j_loc++)
		{
			int j_real = j2 * MEM_UNROLL + j_loc;

			matrix[array_offset_2 + j_real] = peri_row[i2 * BSIZE + j_real];
			matrix[array_offset_3 + j_real] = peri_col[i2 * BSIZE + j_real];
		}

		if (j2 == (BSIZE / MEM_UNROLL) - 1)
		{
			j2 = 0;
			i2++;
			array_offset_1 += matrix_dim;
			array_offset_2 += matrix_dim;
			array_offset_3 += matrix_dim;
		}
		else
		{
			j2++;
		}

		index2++;
	}
}

__attribute__((max_global_work_dim(0)))
__kernel void lud_internal(__global float* restrict matrix, 
                                    int             matrix_dim,
                                    int             offset,
                                    int             i_global,
                                    int             j_global)
{
	float peri_row[BSIZE * BSIZE];
	float peri_col[BSIZE * BSIZE];

	int i1 = 0;
	int j1 = 0;
	int index1 = 0;

	#pragma ivdep
	while (index1 != (BSIZE * BSIZE / MEM_UNROLL))
	{
		#pragma unroll
		for (int j_loc = 0; j_loc < MEM_UNROLL; j_loc++)
		{
			int j_real = j1 * MEM_UNROLL + j_loc;

			peri_row[i1 * BSIZE + j_real] = matrix[matrix_dim * (i1 + offset) + j_real + j_global];
			peri_col[i1 * BSIZE + j_real] = matrix[matrix_dim * (i1 + i_global) + offset + j_real];
		}

		if (j1 == (BSIZE / MEM_UNROLL) - 1)
		{
			j1 = 0;
			i1++;
		}
		else
		{
			j1++;
		}

		index1++;
	}

	int i2 = 0;
	int j2 = 0;
	int index2 = 0;
	// use ivdep since the BB address never repeats
	#pragma ivdep array(matrix)
	while (index2 != (BSIZE * BSIZE / INT_UNROLL))
	{
		index2++;

		#pragma unroll
		for (int j_loc = 0; j_loc < INT_UNROLL; j_loc++)
		{
			float sum = 0.0f;
			int j_real = j2 * INT_UNROLL + j_loc;

			#pragma unroll
			for (int k = 0; k < BSIZE; k++)
			{
				sum += peri_col[BSIZE * i2 + k] * peri_row[BSIZE * k + j_real];
			}
			matrix[(i2 + i_global) * matrix_dim + (j_real + j_global)] -= sum;
		}

		if (j2 == (BSIZE / INT_UNROLL) - 1)
		{
			j2 = 0;
			i2++;
		}
		else
		{
			j2++;
		}
	}
}
