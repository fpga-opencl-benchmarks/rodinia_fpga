#ifndef BSIZE
#if defined(AOCL_BOARD_de5net_a7)
#define BSIZE (16)
#endif
#endif

#include "hotspot_common.h"
#include "../common/opencl_kernel_common.h"

#define SHIFT_REG_COLS (BSIZE + 2)
#define SHIFT_REG_SIZE ((SHIFT_REG_COLS) * 2 + 1)

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

  for (int cb = 0; cb < grid_cols; cb += BSIZE) {
    float shift_reg[SHIFT_REG_SIZE];
    int w_index = cb + ((cb == 0) ? 0 : -1);
    shift_reg[SHIFT_REG_COLS - 1] = temp_src[w_index];
    shift_reg[SHIFT_REG_COLS - 1 + SHIFT_REG_COLS] = temp_src[w_index + grid_cols];
#pragma unroll
    for (int c = 0; c < BSIZE; ++c) {
      shift_reg[c] = temp_src[cb + c];
      shift_reg[c + SHIFT_REG_COLS] = temp_src[cb + c];
    }
    int e_index = cb + ((cb + BSIZE) == grid_cols ? BSIZE - 1 : BSIZE);
    float e = temp_src[e_index];
    shift_reg[SHIFT_REG_COLS - 2] = e;
    shift_reg[SHIFT_REG_COLS - 2 + SHIFT_REG_COLS] = e;
    
    for (int r = 0; r < grid_rows; ++r) {
      int yidx_n = r + ((r == grid_rows - 1) ? 0 : 1);
      //#pragma unroll
      for (int c = 0; c < BSIZE; ++c) {
        int xidx = cb + c;
        int idx = xidx + r * grid_cols;
        int idx_n = xidx + yidx_n * grid_cols;
        int sr_index = SHIFT_REG_COLS;
        int offset_n = SHIFT_REG_COLS;
        int offset_s = - SHIFT_REG_COLS;
        int offset_e = 1;
        int offset_w = -1;
        float nv = temp_src[idx_n];
        shift_reg[sr_index + offset_n] = nv;
        float v = power[idx] +
            (shift_reg[sr_index + offset_n] + shift_reg[sr_index + offset_s]
             - 2.0f * shift_reg[sr_index]) * Ry_1 + 
            (shift_reg[sr_index + offset_e] + shift_reg[sr_index + offset_w]
             - 2.0f * shift_reg[sr_index]) * Rx_1 +
            (AMB_TEMP - shift_reg[sr_index]) * Rz_1;
        float delta = step_div_Cap * v;
        temp_dst[idx] = shift_reg[sr_index] + delta;
#pragma unroll
        for (int i = 0; i < SHIFT_REG_SIZE-1; ++i) {
          shift_reg[i] = shift_reg[i+1];
        }

        // NOTE: This block can be moved out from the loop by index c,
        // but doing so prevents the compiler to pipeline the
        // loop. This looks inefficient but hopefully okay since a
        // conditional branch should not be a big performance problem
        // on FPGA. 
        if (c == BSIZE - 1) {
          if (r < grid_rows - 1) {
#pragma unroll
            for (int i = 0; i < SHIFT_REG_SIZE-2; ++i) {
              shift_reg[i] = shift_reg[i+2];
            }
            int xidx_e = cb + ((cb + BSIZE) == grid_cols ? BSIZE - 1 : BSIZE);
            shift_reg[SHIFT_REG_SIZE-3] = temp_src[xidx_e + yidx_n * grid_cols];
            int xidx_w = cb + ((cb == 0) ? 0 : -1);
            int yidx_nn = (yidx_n == grid_rows - 1) ? grid_rows - 1 : yidx_n + 1;
            shift_reg[SHIFT_REG_SIZE-2] = temp_src[xidx_w + yidx_nn * grid_cols];
          }
        }
      }
    }
  }
}
