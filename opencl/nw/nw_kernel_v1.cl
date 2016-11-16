#include "../common/opencl_kernel_common.h"

int maximum(int a,
	    int b,
	    int c)
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
nw_kernel1(__global int* reference, 
           __global int* input_itemsets,
           int max_cols,
           int penalty) 
{
  for (int j = 1; j < max_cols-1; ++j) {
    for (int i = 1; i < max_cols-1; ++i) {
      int index = j * max_cols + i;
      input_itemsets[index]= maximum(
          input_itemsets[index-1-max_cols]+ reference[index], 
          input_itemsets[index-1]         - penalty, 
          input_itemsets[index-max_cols]  - penalty);
    }
  }
}
