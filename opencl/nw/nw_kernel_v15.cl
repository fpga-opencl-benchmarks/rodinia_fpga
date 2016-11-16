// Override the default block size for some devices
#ifndef BSIZE
#if defined(AOCL_BOARD_de5net_a7)
#define BSIZE (64)
#endif
#endif

#include "../common/opencl_kernel_common.h"

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
nw_kernel1(__global const int * RESTRICT reference, 
           __global const int * RESTRICT input_itemsets_h,
           __global const int * RESTRICT input_itemsets_v,
           __global int * RESTRICT output_itemsets,           
           __global int * RESTRICT output_itemsets_v,
           __global int * RESTRICT output_itemsets_h,
           int max_cols,
           int penalty,
           int bx) {
  int max_rows = max_cols;
  int base = BSIZE * bx + 1;
  int row_max = (max_rows - 1) / 2;

  int sr[BSIZE];

#pragma unroll
  for (int i = 0; i < BSIZE; ++i) {
    sr[i] = input_itemsets_h[base + i];
  }

  for (int j = 1; j < row_max; ++j) {
    int diag = j == 1 ? input_itemsets_h[base - 1] : input_itemsets_v[j-1];
    int left = input_itemsets_v[j];

#pragma unroll
    for (int i = 0; i < BSIZE; ++i) {
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
      if (i == BSIZE - 1) {
        output_itemsets_v[j] =  v;
      }
    }
  }

#pragma unroll
  for (int i = 0; i < BSIZE; ++i) {
    output_itemsets_h[base + i - 1] = sr[i];
  }
  
}

__kernel void 
nw_kernel2(__global const int * RESTRICT reference, 
           __global const int * RESTRICT input_itemsets_h,
           __global const int * RESTRICT input_itemsets_v,
           __global int * RESTRICT output_itemsets,           
           __global int * RESTRICT output_itemsets_v,
           int max_cols,
           int penalty,
           int bx) {
  int max_rows = max_cols;
  int base = BSIZE * bx + 1;
  int row_max = (max_rows - 1) / 2;

  int sr[BSIZE];

#pragma unroll
  for (int i = 0; i < BSIZE; ++i) {
    sr[i] = input_itemsets_h[base + i - 1];
  }

  for (int j = row_max; j < max_rows - 1; ++j) {
    int diag = j == 1 ? input_itemsets_h[base - 1 - 1] : input_itemsets_v[j-1];
    int left = input_itemsets_v[j];

#pragma unroll
    for (int i = 0; i < BSIZE; ++i) {
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
      if (i == BSIZE - 1) {
        output_itemsets_v[j] =  v;
      }
    }
  }
}
