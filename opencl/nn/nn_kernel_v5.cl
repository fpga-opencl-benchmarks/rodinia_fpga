//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#include "../common/opencl_kernel_common.h"

typedef struct latLong
	{
		float lat;
		float lng;
	} LatLong;

__kernel void NearestNeighbor(__global LatLong* RESTRICT d_locations,
			      __global float*   RESTRICT d_distances,
			      const    int               numRecords,
			      const    float             lat,
			      const    float             lng)
{
	int i;
	
	#pragma unroll 64
	for (i=0; i<numRecords; i++)
	{
		d_distances[i] = (float)sqrt( (lat - d_locations[i].lat) * (lat - d_locations[i].lat) + (lng - d_locations[i].lng) * (lng - d_locations[i].lng) );
	}
} 
