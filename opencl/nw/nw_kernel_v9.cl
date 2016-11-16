// Override the default block size for some devices
#ifndef BSIZE
#if defined(AOCL_BOARD_de5net_a7)
#define BSIZE (128)
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
           int mc1,
           int penalty,
           int bx) {
  int mr1 = mc1;
  int base = BSIZE * bx;

  int sr[BSIZE];

#pragma unroll
  for (int i = 0; i < BSIZE; ++i) {
    sr[i] = input_itemsets_h[base + i];
  }

  for (int j = 0; j < mr1 - 1; ++j) {
    int diag = j == 0 ? (bx == 0 ? 0 : input_itemsets_h[base-1]) :
        input_itemsets_v[j-1];
    int left = input_itemsets_v[j];

#pragma unroll
    for (int i = 0; i < BSIZE; ++i) {
      int index = base + i + mc1 * j;
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
