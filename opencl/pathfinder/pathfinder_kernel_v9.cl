#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (16)
#endif
#ifndef XS
#define XS (4)
#endif
#define BLOCK_SIZE2 ((BLOCK_SIZE) * (XS))
#define DIAG_SIZE 100
#define SW_SIZE ((DIAG_SIZE) * (BLOCK_SIZE2) * 2 + (BLOCK_SIZE2))

#define OFFSET(x, y) ((x) + (y) * cols)

__kernel void dynproc_kernel (__global int* restrict wall,
                              __global int* restrict results,
                              int  cols,
                              int  rows) {

  int diag = -1;
  int x = 0, y = 0, xs = -1;
  int diag_idx = 1;
  
  int sw[SW_SIZE];

#pragma unroll
  for (int i = 0; i < SW_SIZE; ++i) {
    sw[i] = INT_MAX;
  }
  
  do {
    ++xs;
    if (xs == XS) {
      xs = 0;
      --diag_idx;
      if (diag_idx > 0) {
        x -= BLOCK_SIZE2;
        ++y;
      } else {
        // next diagonal line
        diag_idx = DIAG_SIZE;
        ++diag;
        x = diag * BLOCK_SIZE2;
        y = 0;
        //printf("x, y = %d, %d\n", x, y);
      }
    }
    //printf("x, y = %d, %d\n", x, y);
    int xxs = x + xs * BLOCK_SIZE;
    int va[BLOCK_SIZE];
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      int xxsi = xxs + i;
      if (xxsi >= 0 && xxsi < cols) {    
        //printf("%d: x, y = %d, %d\n", dbg_idx, x, y);
        va[i] = wall[OFFSET(xxsi, y)];
        //printf("w [%d, %d] %d\n", x + i, y, va[i]);
        int center =  SW_SIZE - (DIAG_SIZE * BLOCK_SIZE2) - BLOCK_SIZE2 + i;        
#if 0        
        int right =  (xs == XS - 1 && i == BLOCK_SIZE - 1) ?
            SW_SIZE - BLOCK_SIZE2 * 2 + BLOCK_SIZE : center + 1;
        int left = (xs == 0 && i == 0) ? BLOCK_SIZE2 - 1 : center - 1;
        if (y != 0) {
          va[i] += MIN(MIN(sw[left], sw[center]), sw[right]);
          //printf("w [%d, %d] %d, %d\n", x + i, y, va[i], sw[left]);
        }
#endif
        if (y != 0) {
          int s_c = sw[center];
          int s_r;
          if (xs == XS - 1 && i == BLOCK_SIZE - 1) {
            s_r = sw[SW_SIZE - BLOCK_SIZE2 * 2 + BLOCK_SIZE];
          } else {
            s_r = sw[center + 1];
          }
          int s_l;
          if (xs == 0 && i == 0) {
            s_l = sw[BLOCK_SIZE2 - 1];
          } else {
            s_l = sw[center - 1];
          }
          va[i] += MIN(MIN(s_l, s_c), s_r);
        }
      } else {
        va[i] = INT_MAX;
      }
    }
#pragma unroll
    for (int i = 0; i < SW_SIZE - BLOCK_SIZE; ++i) {
      sw[i] = sw[i + BLOCK_SIZE];
    }
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      sw[SW_SIZE - BLOCK_SIZE + i] = va[i];
      //printf("[%d, %d] %d\n", x + i, y, va[i]);
    }
#pragma unroll
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      if (xxs + i >= 0 && xxs + i < cols) results[xxs + i] = va[i];
    }
  } while (x + (xs + 1) * BLOCK_SIZE < cols || y != rows - 1);
}




