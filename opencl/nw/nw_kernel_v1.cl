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

__kernel void nw_kernel1(__global int* restrict reference, 
                         __global int* restrict input_itemsets,
                                  int           dim,
                                  int           penalty)
{
	for (int j = 1; j < dim - 1; ++j)
	{
		for (int i = 1; i < dim - 1; ++i)
		{
			int index = j * dim + i;
			input_itemsets[index]= maximum(input_itemsets[index - 1 - dim] + reference[index],
									 input_itemsets[index - 1]       - penalty,
									 input_itemsets[index - dim]     - penalty);
		}
	}
}
