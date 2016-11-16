// Override the default block size for some devices
#ifndef BSIZE
#if defined(AOCL_BOARD_de5net_a7)
#define BSIZE (64)
#endif
#endif

#include "../common/opencl_kernel_common.h"

#pragma OPENCL EXTENSION cl_altera_channels : enable

#define CDEPTH __attribute__((depth(128)))
//#define CDEPTH 
channel int boundary[3] CDEPTH;

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

#undef CH_IN_IDX
#define CH_IN_IDX 0
#undef CH_OUT_IDX 
#define CH_OUT_IDX 1

__kernel void 
nw_kernel1(__global const int * RESTRICT reference, 
           __global const int * RESTRICT input_itemsets_h,
           __global const int * RESTRICT input_itemsets_v,
           __global int * RESTRICT output_itemsets,           
           int max_cols,
           int penalty,
           int bx) {
  int max_rows = max_cols;
  int base = BSIZE * bx + 1;

  int sr[BSIZE + 1];

#pragma unroll
  for (int i = 0; i < BSIZE + 1; ++i) {
    sr[i] = input_itemsets_h[base + i - 1];
  }

  for (int j = 1; j < max_rows - 1; ++j) {
    int diag = sr[0];
    //int left = input_itemsets_v[j];
    int left = bx == 0 ? input_itemsets_v[j]:
        read_channel_altera(boundary[CH_IN_IDX]);
    sr[0] = left;

    int b;
#pragma unroll
    for (int i = 0; i < BSIZE; ++i) {
      int index = base + i + max_cols * j;
      int above = sr[i+1];
      int v = 
          maximum(
              diag + reference[index], 
              left - penalty,
              above - penalty);
      diag = above;
      left = v;
      sr[i+1] = v;
      output_itemsets[index] = v;
      if (i == BSIZE - 1) b = v;
    }
    //output_itemsets_v[j] =  v;
    write_channel_altera(boundary[CH_OUT_IDX], b);
  }
}

#undef CH_IN_IDX
#define CH_IN_IDX 1
#undef CH_OUT_IDX 
#define CH_OUT_IDX 2

__kernel void 
nw_kernel2(__global const int * RESTRICT reference, 
           __global const int * RESTRICT input_itemsets_h,
           __global int * RESTRICT output_itemsets,           
           int max_cols,
           int penalty,
           int bx) {
  int max_rows = max_cols;
  int base = BSIZE * bx + 1;

  int sr[BSIZE + 1];

#pragma unroll
  for (int i = 0; i < BSIZE + 1; ++i) {
    sr[i] = input_itemsets_h[base + i - 1];
  }

  for (int j = 1; j < max_rows - 1; ++j) {
    int diag = sr[0];
    //int left = input_itemsets_v[j];
    int left = read_channel_altera(boundary[CH_IN_IDX]);
    sr[0] = left;

    int b;
#pragma unroll
    for (int i = 0; i < BSIZE; ++i) {
      int index = base + i + max_cols * j;
      int above = sr[i+1];
      int v = 
          maximum(
              diag + reference[index], 
              left - penalty,
              above - penalty);
      diag = above;
      left = v;
      sr[i+1] = v;
      output_itemsets[index] = v;
      if (i == BSIZE - 1) b = v;
    }
    //output_itemsets_v[j] =  b;
    write_channel_altera(boundary[CH_OUT_IDX], b);
  }
}

#undef CH_IN_IDX
#define CH_IN_IDX 2
#undef CH_OUT_IDX 
#define CH_OUT_IDX 0

__kernel void 
nw_kernel3(__global const int * RESTRICT reference, 
           __global const int * RESTRICT input_itemsets_h,
           __global int * RESTRICT output_itemsets,           
           int max_cols,
           int penalty,
           int bx) {
  int max_rows = max_cols;
  int base = BSIZE * bx + 1;

  int sr[BSIZE + 1];

#pragma unroll
  for (int i = 0; i < BSIZE + 1; ++i) {
    sr[i] = input_itemsets_h[base + i - 1];
  }

  for (int j = 1; j < max_rows - 1; ++j) {
    int diag = sr[0];
    //int left = input_itemsets_v[j];
    int left = read_channel_altera(boundary[CH_IN_IDX]);
    sr[0] = left;

    int b;
#pragma unroll
    for (int i = 0; i < BSIZE; ++i) {
      int index = base + i + max_cols * j;
      int above = sr[i+1];
      int v = 
          maximum(
              diag + reference[index], 
              left - penalty,
              above - penalty);
      diag = above;
      left = v;
      sr[i+1] = v;
      output_itemsets[index] = v;
      if (i == BSIZE - 1) b = v;
    }
    //output_itemsets_v[j] =  b;
    write_channel_altera(boundary[CH_OUT_IDX], b);
  }
}
#if 0
#undef CH_IN_IDX
#define CH_IN_IDX 3
#undef CH_OUT_IDX
#define CH_OUT_IDX 0

__kernel void 
nw_kernel4(__global const int * RESTRICT reference, 
           __global const int * RESTRICT input_itemsets_h,
           __global int * RESTRICT output_itemsets,           
           int max_cols,
           int penalty,
           int bx) {
  int max_rows = max_cols;
  int base = BSIZE * bx + 1;

  int sr[BSIZE + 1];

#pragma unroll
  for (int i = 0; i < BSIZE + 1; ++i) {
    sr[i] = input_itemsets_h[base + i - 1];
  }

  for (int j = 1; j < max_rows - 1; ++j) {
    int diag = sr[0];
    //int left = input_itemsets_v[j];
    int left = read_channel_altera(boundary[CH_IN_IDX]);
    sr[0] = left;

    int b;
#pragma unroll
    for (int i = 0; i < BSIZE; ++i) {
      int index = base + i + max_cols * j;
      int above = sr[i+1];
      int v = 
          maximum(
              diag + reference[index], 
              left - penalty,
              above - penalty);
      diag = above;
      left = v;
      sr[i+1] = v;
      output_itemsets[index] = v;
      if (i == BSIZE - 1) b = v;
    }
    //output_itemsets_v[j] =  b;
    write_channel_altera(boundary[CH_OUT_IDX], b);
  }
}

#endif
