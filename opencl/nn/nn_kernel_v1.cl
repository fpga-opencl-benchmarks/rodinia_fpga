//#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct latLong
	{
		float lat;
		float lng;
	} LatLong;

__kernel void NearestNeighbor(__global LatLong* restrict d_locations,
			      __global float  * restrict d_distances,
			      const    int     numRecords,
			      const    float   lat,
			      const    float   lng)
{
	int i;
	for (i=0; i<numRecords; i++)
	{
		d_distances[i] = (float)sqrt( (lat - d_locations[i].lat) * (lat - d_locations[i].lat) + (lng - d_locations[i].lng) * (lng - d_locations[i].lng) );
	}
} 
