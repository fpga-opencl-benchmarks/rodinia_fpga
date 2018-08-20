#include "work_group_size.h"

#define SCORE(i, j) input_itemsets_l[j + i * (BSIZE+1)]
#define REF(i, j)   reference_l[j + i * BSIZE]

int maximum(int a, int b, int c)
{
	int k;
	if(a <= b)
		k = b;
	else 
		k = a;

	if(k <= c)
		return(c);
	else
		return(k);
}

__kernel void nw_kernel1(__global int* restrict reference_d, 
                         __global int* restrict input_itemsets_d, 
                                  int           cols,
                                  int           penalty,
                                  int           blk,
                                  int           offset_r,
                                  int           offset_c)
{
	// Block index
	int bx = get_group_id(0);	

	// Thread index
	int tx = get_local_id(0);

	// Base elements
	int base = offset_r * cols + offset_c;

	// Local buffers
	__local int input_itemsets_l[(BSIZE + 1) * (BSIZE + 1)];
	__local int reference_l[BSIZE * BSIZE];

	int b_index_x = bx;
	int b_index_y = blk - 1 - bx;

	int index    = base + cols * BSIZE * b_index_y + BSIZE * b_index_x + tx + cols + 1;
	int index_n  = base + cols * BSIZE * b_index_y + BSIZE * b_index_x + tx + 1;
	int index_w  = base + cols * BSIZE * b_index_y + BSIZE * b_index_x + cols;
	int index_nw = base + cols * BSIZE * b_index_y + BSIZE * b_index_x;

	if (tx == 0)
	{
		SCORE(tx, 0) = input_itemsets_d[index_nw + tx];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int ty = 0 ; ty < BSIZE ; ty++)
	{
		REF(ty, tx) = reference_d[index + cols * ty];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE((tx + 1), 0) = input_itemsets_d[index_w + cols * tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE(0, (tx + 1)) = input_itemsets_d[index_n];

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int m = 0 ; m < BSIZE ; m++)
	{
		if (tx <= m)
		{
			int t_index_x = tx + 1;
			int t_index_y = m - tx + 1;

			SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1))     ,
										    SCORE((t_index_y)  , (t_index_x-1)) - (penalty)        , SCORE((t_index_y-1),
										    (t_index_x)) - (penalty));
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int m = BSIZE - 2 ; m >=0 ; m--)
	{
		if (tx <= m)
		{
			int t_index_x = tx + BSIZE - m ;
			int t_index_y = BSIZE - tx;

			SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1))     ,
		                                            SCORE((t_index_y)  , (t_index_x-1)) - (penalty)        , SCORE((t_index_y-1),
										    (t_index_x)) - (penalty));
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (int ty = 0 ; ty < BSIZE ; ty++)
	{
		input_itemsets_d[index + cols * ty] = SCORE((ty+1), (tx+1));
	}

	return;
}

__kernel void nw_kernel2(__global int* restrict reference_d, 
                         __global int* restrict input_itemsets_d, 
                                  int           cols,
                                  int           penalty,
                                  int           blk,
                                  int           block_width,
                                  int           offset_r,
                                  int           offset_c)
{
	int bx = get_group_id(0);	

	// Thread index
	int tx = get_local_id(0);
	
	// Base elements
	int base = offset_r * cols + offset_c;
	
	// Local buffers
	__local int input_itemsets_l[(BSIZE + 1) * (BSIZE + 1)];
	__local int reference_l[BSIZE * BSIZE];

	int b_index_x = bx + block_width - blk;
	int b_index_y = block_width - bx -1;
	
	int index    = base + cols * BSIZE * b_index_y + BSIZE * b_index_x + tx + cols + 1;
	int index_n  = base + cols * BSIZE * b_index_y + BSIZE * b_index_x + tx + 1;
	int index_w  = base + cols * BSIZE * b_index_y + BSIZE * b_index_x + cols;
	int index_nw = base + cols * BSIZE * b_index_y + BSIZE * b_index_x;
	
	if (tx == 0)
	{
		SCORE(tx, 0) = input_itemsets_d[index_nw];
	}

	for (int ty = 0 ; ty < BSIZE ; ty++)
	{
		REF(ty, tx) = reference_d[index + cols * ty];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE((tx + 1), 0) = input_itemsets_d[index_w + cols * tx];

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE(0, (tx + 1)) = input_itemsets_d[index_n];

	barrier(CLK_LOCAL_MEM_FENCE);

	for(int m = 0 ; m < BSIZE ; m++)
	{
		if (tx <= m)
		{
			int t_index_x = tx + 1;
			int t_index_y = m - tx + 1;

			SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1))     ,
		                                            SCORE((t_index_y)  , (t_index_x-1)) - (penalty)        , SCORE((t_index_y-1),
										    (t_index_x)) - (penalty));
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for(int m = BSIZE - 2; m >=0; m--)
	{
		if (tx <= m)
		{
			int t_index_x = tx + BSIZE - m;
			int t_index_y = BSIZE - tx;

			SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1))     ,
		                                            SCORE((t_index_y)  , (t_index_x-1)) - (penalty)        , SCORE((t_index_y-1),
										    (t_index_x)) - (penalty));
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	for (int ty = 0 ; ty < BSIZE ; ty++)
	{
		input_itemsets_d[index + ty * cols] = SCORE((ty+1), (tx+1));
	}

	return;
}