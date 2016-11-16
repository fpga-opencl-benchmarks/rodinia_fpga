#ifndef BSIZE
#if defined(AOCL_BOARD_de5net_a7)
#define BSIZE (32)
#endif
#endif

#include "hotspot_common.h"
#include "../common/opencl_kernel_common.h"

#ifndef SSIZE
#define SSIZE (16)
#endif

#define SW_COLS (BSIZE)
#define SW_BASE_SIZE ((SW_COLS) * 2)
#define SW_SIZE (SW_BASE_SIZE + SSIZE)

__kernel void hotspot(__global const float * RESTRICT power,   //power input
                      __global const float * RESTRICT temp_src,  //temperature input/output
                      __global float * RESTRICT temp_dst,    //temperature input/output
                      int grid_cols,  //Col of grid
                      int grid_rows,  //Row of grid
                      float Cap,      //Capacitance
                      float Rx, float Ry, float Rz, 
                      float step) {
  float step_div_Cap = step/Cap;
  float Rx_1 = 1 / Rx;
  float Ry_1 = 1 / Ry;
  float Rz_1 = 1 / Rz;

  for (int cb = 0; cb < grid_cols; cb += BSIZE) {
    
    float sw[SW_SIZE];

    // initialize
#pragma unroll
    for (int i = 0; i < SW_SIZE; ++i) {
      sw[i] = 0.0f;
    }

#pragma unroll
    for (int c = 0; c < BSIZE; ++c) {
      sw[SSIZE + c] = temp_src[cb + c];
      sw[SSIZE + c + SW_COLS] = temp_src[cb + c];
    }
    
    int x = 0, y = 0;
    do {
      int gx = cb + x;
      int comp_offset = gx + y * grid_cols;      
      int read_offset = comp_offset +
          (y < grid_rows - 1 ? grid_cols : 0);
      
      // shift
#pragma unroll
      for (int i = 0; i < SW_BASE_SIZE; ++i) {
        sw[i] = sw[i + SSIZE];
      }

      // read new values
#pragma unroll
      for (int i = 0; i < SSIZE; ++i) {
        sw[SW_BASE_SIZE + i] = temp_src[read_offset + i];
      }
      
      float value[SSIZE];
      int sw_offset = SW_COLS;
      int sw_offset_n = sw_offset + SW_COLS;
      int sw_offset_s = sw_offset - SW_COLS;
      int sw_offset_e = sw_offset + 1;
      int sw_offset_w = sw_offset - 1;

#pragma unroll
      for (int i = 0; i < SSIZE; ++i) {
        float w;
        if (i == 0 && gx == 0) {
          w = sw[sw_offset];
        } else if (i ==0 && x == 0) {
          w = temp_src[comp_offset - 1];
        } else {
          w = sw[sw_offset_w + i];
        }
        float e;
        if (i == SSIZE - 1 && gx + SSIZE == grid_cols) {
          e = sw[sw_offset + i];
        } else if (i == SSIZE - 1 && x + SSIZE == BSIZE) {
          e = temp_src[comp_offset + SSIZE];
        } else {
          e = sw[sw_offset_e + i];
        }
        float v = power[comp_offset + i] +
            (e + w - 2.0f * sw[sw_offset + i]) * Rx_1 +
            (sw[sw_offset_n + i] + sw[sw_offset_s + i] -
             2.0f * sw[sw_offset + i]) * Ry_1 +
            (AMB_TEMP - sw[sw_offset + i]) * Rz_1;
        float delta = step_div_Cap * v;
        value[i] = sw[sw_offset + i] + delta;
      }

#pragma unroll
      for (int i = 0; i < SSIZE; ++i) {
        temp_dst[comp_offset + i] = value[i];
      }

      x = x < BSIZE - SSIZE ? x + SSIZE : 0;
      y = x == 0 ? y + 1 : y;
    } while (y < grid_rows);
  }
}
