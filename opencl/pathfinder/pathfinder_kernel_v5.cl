#define MIN(a, b) ((a)<=(b) ? (a) : (b))

#define BLOCK_SIZE 16
#define DIAG_SIZE 100
#define SW_SIZE ((DIAG_SIZE) * (BLOCK_SIZE) * 2 + (BLOCK_SIZE))

#define OFFSET(x, y) ((x) + (y) * cols)

__kernel void dynproc_kernel (__global int* restrict wall,
                              __global int* restrict results,
                              int  cols,
                              int  rows) {

  int diag = -1;
  int x = 0, y = 0;
  int diag_idx = 1;
  int dbg_idx = -1;

  int sw[SW_SIZE];

#pragma unroll
  for (int i = 0; i < SW_SIZE; ++i) {
    sw[i] = INT_MAX;
  }
  
  do {
    --diag_idx;
    ++dbg_idx;
    if (diag_idx > 0) {
      x -= BLOCK_SIZE;
      ++y;
    } else {
      // next diagonal line
      diag_idx = DIAG_SIZE;
      ++diag;
      x = diag * BLOCK_SIZE;
      y = 0;
      //printf("x, y = %d, %d\n", x, y);
    }
    //printf("x, y = %d, %d\n", x, y);    
    int va[BLOCK_SIZE];
    if (x >= 0 && x < cols) {    
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        //printf("%d: x, y = %d, %d\n", dbg_idx, x, y);
        va[i] = wall[OFFSET(x + i, y)];
        //printf("w [%d, %d] %d\n", x + i, y, va[i]);
        int center =  SW_SIZE - (DIAG_SIZE * BLOCK_SIZE) - BLOCK_SIZE + i;
        int right =  i == BLOCK_SIZE - 1 ? SW_SIZE - BLOCK_SIZE : center + 1;
        int left = i == 0 ? BLOCK_SIZE - 1 : center - 1;
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
    if (x >= 0 && x < cols && y == rows - 1) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        results[x + i] = va[i];
      }
    }
  } while (x != cols - BLOCK_SIZE || y != rows - 1);
}




