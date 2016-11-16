//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#include "../common/opencl_kernel_common.h"

typedef struct latLong
	{
		float lat;
		float lng;
	} LatLong;

#ifndef SIMD_LANES
	#define SIMD_LANES 16
#endif
	
#ifndef COMPUTE_UNITS
	#define COMPUTE_UNITS 3
#endif

__attribute__((reqd_work_group_size(64,1,1)))
__attribute__((num_simd_work_items(SIMD_LANES)))
__attribute__((num_compute_units(COMPUTE_UNITS)))
__kernel void NearestNeighbor(__global LatLong* RESTRICT d_locations,
			      __global float*   RESTRICT d_distances,
			      const    int               numRecords,
			      const    float             lat,
			      const    float             lng)
{
	int globalId = get_global_id(0);
  
	if (globalId < numRecords)
	{
		__global LatLong *latLong = d_locations + globalId;
		__global float *dist = d_distances + globalId;
		*dist = (float)sqrt( (lat - latLong->lat) * (lat - latLong->lat) + (lng - latLong->lng) * (lng - latLong->lng) );
	}
}