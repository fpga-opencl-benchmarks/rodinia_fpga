#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define BLOCK_SIZE (10)
#define XS (5)
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
    if (xxs >= 0 && xxs < cols) {    
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        //printf("%d: x, y = %d, %d\n", dbg_idx, x, y);
        va[i] = wall[OFFSET(xxs + i, y)];
        //printf("w [%d, %d] %d\n", x + i, y, va[i]);
        int center =  SW_SIZE - (DIAG_SIZE * BLOCK_SIZE2) - BLOCK_SIZE2 + i;
        int right =  (xs == XS - 1 && i == BLOCK_SIZE - 1) ?
            SW_SIZE - BLOCK_SIZE2 * 2 + BLOCK_SIZE : center + 1;
        int left = (xs == 0 && i == 0) ? BLOCK_SIZE2 - 1 : center - 1;
        if (y != 0) {
          va[i] += MIN(MIN(sw[left], sw[center]), sw[right]);
          //printf("w [%d, %d] %d, %d\n", x + i, y, va[i], sw[left]);
        }
      }
    } else {
      //printf("%d: x, y = %d, %d (out of range)\n", dbg_idx, x, y);
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
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
    if (xxs >= 0 && xxs < cols && y == rows - 1) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        results[xxs + i] = va[i];
      }
    }
  } while (x + xs * BLOCK_SIZE != cols - BLOCK_SIZE || y != rows - 1);
}




