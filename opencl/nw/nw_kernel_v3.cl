inline int maximum(int a, int b, int c)
{
	int k;
	if(a <= b)
		k = b;
	else
		k = a;

	if(k <= c)
		return(c);
	else
		return(k);
}

__attribute__((max_global_work_dim(0)))
__kernel void nw_kernel1(__global int* restrict reference, 
                         __global int* restrict input_itemsets,
                                  int           dim,
                                  int           penalty) 
{
	for (int j = 1; j < dim - 1; ++j)
	{
		int backup = input_itemsets[j * dim];

		#pragma ivdep array(input_itemsets)
		for (int i = 1; i < dim - 1; ++i)
		{
			int index = j * dim + i;
			input_itemsets[index] = backup = maximum(input_itemsets[index - 1 - dim] + reference[index],
			                                         backup                          - penalty,
			                                         input_itemsets[index - dim]     - penalty);
		}
	}
}
