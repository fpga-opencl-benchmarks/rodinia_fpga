#include "../common/opencl_kernel_common.h"
#include "work_group_size.h"

#define SCORE(i, j) input_itemsets_l[j + i * (BLOCK_SIZE+1)]
#define REF(i, j)   reference_l[j + i * BLOCK_SIZE]

int maximum(int a,
	    int b,
	    int c)
{
	int k;
	if( a <= b )
		k = b;
	else
		k = a;

	if( k <=c )
		return(c);
	else
		return(k);
}

__kernel void 
nw_kernel1(__global int* RESTRICT reference, 
           __global int volatile * RESTRICT input_itemsets,
           __global int* RESTRICT output_itemsets,           
           int max_cols,
           int penalty,
           int bx, int by) 
{
  int base = BLOCK_SIZE * bx + 1 + max_cols * (BLOCK_SIZE * by + 1);
  int sr[BLOCK_SIZE];
  
  #pragma unroll
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    sr[i] = input_itemsets[base - max_cols + i];
  }
  
  for (int j = 0; j < BLOCK_SIZE; ++j) {
    int diag = input_itemsets[base + max_cols * j - 1 - max_cols];
    int left = input_itemsets[base + max_cols * j - 1];

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      int index = base + i + max_cols * j;
      int above = sr[i];
      int v = 
          maximum(
          diag + reference[index], 
          left - penalty,
          above - penalty);
      diag = above;
      left = v;
      sr[i] = v;
      output_itemsets[index] = v;
    }
  }
}

