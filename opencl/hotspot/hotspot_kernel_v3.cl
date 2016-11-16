#ifndef BSIZE
#if defined(AOCL_BOARD_de5net_a7)
#define BSIZE (16)
#endif
#endif

#include "hotspot_common.h"
#include "../common/opencl_kernel_common.h"

__kernel void hotspot(  __global const float * RESTRICT power,   //power input
                        __global const float * RESTRICT temp_src,    //temperature input/output
                        __global float * RESTRICT temp_dst,    //temperature input/output
                        int grid_cols,  //Col of grid
                        int grid_rows,  //Row of grid
                        float Cap,      //Capacitance
                        float Rx, 
                        float Ry, 
                        float Rz, 
                        float step) {
  float step_div_Cap = step/Cap;
  float Rx_1 = 1 / Rx;
  float Ry_1 = 1 / Ry;
  float Rz_1 = 1 / Rz;

  // Since grid_cols is not constant, compilers may not unroll the
  // loop. To unroll the inner loop, block the loop by BSIZE.
  for (int r = 0; r < grid_rows; ++r) {        
    for (int cb = 0; cb < grid_cols; cb += BSIZE) {
#pragma unroll
      for (int c = 0; c < BSIZE; ++c) {
        int cg = cb + c;
        int index = cg + r * grid_cols;
        int offset_n = (r == grid_rows - 1) ? 0 : grid_cols;
        int offset_s = (r == 0) ? 0 : - grid_cols;
        int offset_e = (cg == grid_cols - 1) ? 0 : 1;
        int offset_w = (cg == 0) ? 0 : -1;
        float v = power[index] +
            (temp_src[index + offset_n] + temp_src[index + offset_s]
             - 2.0f * temp_src[index]) * Ry_1 + 
            (temp_src[index + offset_e] + temp_src[index + offset_w]
             - 2.0f * temp_src[index]) * Rx_1 +
            (AMB_TEMP - temp_src[index]) * Rz_1;
        float delta = step_div_Cap * v;
        temp_dst[index] = temp_src[index] + delta;
      }
    }
  }
}
