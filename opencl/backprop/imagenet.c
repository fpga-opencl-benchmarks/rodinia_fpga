
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"

extern int layer_size;

void load(BPNN *net)
//BPNN *net;
{
	float *units;
	int nr, i, k;

	nr = layer_size;
	units = net->input_units;

	k = 1;
	for (i = 0; i < nr; i++) {
		units[k] = (float) rand()/RAND_MAX ;
		k++;
	}
}
