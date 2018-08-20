#include "../common/common.h"

#define AA(i,j) m[offset * matrix_dim + i * matrix_dim + j + offset]
#define BB(i,j) m[i * matrix_dim + j]


__kernel void lud_diagonal(__global float* restrict m, 
                                    int             matrix_dim, 
                                    int             offset)
{ 
	int i, j, k;

	for (i = 0; i < BSIZE; i++)
	{
		// ivdep is added to avoid false dependency on the AA access
		// k is always smaller than both i and j; hence, AA(i, index) or AA(index, j) will never become equal to AA(i, j)
		#pragma ivdep array(m)
		for (j = i; j < BSIZE; j++)
		{
			for (k = 0; k < i; k++)
			{
				AA(i, j) = AA(i, j) - AA(i, k) * AA(k, j);
			}
		}
   
		float temp = 1.f / AA(i,i);
		// ivdep is added to avoid false dependency on the AA access
		// k is always smaller than both i and j; hence, AA(i, index) or AA(index, j) will never become equal to AA(i, j)
		#pragma ivdep array(m)
		for (j = i + 1; j < BSIZE; j++)
		{
			for (k = 0; k < i; k++)
			{
				AA(j, i) = AA(j, i) - AA(j, k) * AA(k, i);
			}
			AA(j, i) = AA(j, i) * temp;
		}
	}
}

__kernel void lud_perimeter(__global float* restrict m, 
                                     int             matrix_dim, 
                                     int             offset)
{
	int chunk_idx, chunk_num;

	chunk_num = ((matrix_dim - offset) / BSIZE) - 1;

	for (chunk_idx = 0; chunk_idx < chunk_num; chunk_idx++)
	{
		int i, j, k, i_global, j_global, i_here, j_here;
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
				sum = 0.0f;
				for (k = 0; k < i; k++)
				{
					sum += temp[BSIZE * i + k] * BB((i_global + k), (j_global + j));
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
				sum = 0.0f;
				for (k = 0; k < j; k++)
				{
					sum += BB((i_global + i), (j_global + k)) * temp[BSIZE * k + j];
				}

				i_here = i_global + i;
				j_here = j_global + j;
				BB(i_here, j_here) = (BB(i_here, j_here) - sum) / AA(j, j);
			}
		}
	}
}

__kernel void lud_internal(__global float* restrict m, 
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

		// use ivdep since the BB address never repeats
		#pragma ivdep array(m)
		for (i = 0; i < BSIZE; i++)
		{
			for (j = 0; j < BSIZE; j++)
			{
				float sum = 0.0f;
				for (k = 0; k < BSIZE; k++)
				{
					sum += temp_left[BSIZE * i + k] * temp_top[BSIZE * k + j];
				}
				BB((i + i_global), (j + j_global)) -= sum;
			}
		}
	}
}
