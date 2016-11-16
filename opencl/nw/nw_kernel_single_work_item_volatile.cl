// Override the default block size for some devices
#ifndef BLOCK_SIZE
#if defined(AOCL_BOARD_de5net_a7)
#define BLOCK_SIZE (64)
#warning blocksize
#endif
#endif

#include "../common/opencl_kernel_common.h"
#include "work_group_size.h"

int maximum(int a, int b, int c)
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
nw_kernel1(__global int * RESTRICT reference, 
           __global int volatile * RESTRICT input_itemsets,
           __global int * RESTRICT output_itemsets,           
           int max_cols,
           int penalty,
           int bx) {
  int max_rows = max_cols;
  int base = BLOCK_SIZE * bx + 1;

  int sr[BLOCK_SIZE];
  
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE; ++i) {
    sr[i] = input_itemsets[base + i];
  }

  for (int j = 1; j < max_rows - 1; ++j) {
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
