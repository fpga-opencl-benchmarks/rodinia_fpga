#define FADD_LATENCY 8

#ifndef UNROLL
	#define UNROLL 10
#endif

#if UNROLL == 1
	#define REDUCTION_LATENCY FADD_LATENCY
#else
	#define REDUCTION_LATENCY (UNROLL/2)*FADD_LATENCY
#endif

//========================================================================================================================================================================================================200
//	INCLUDE/DEFINE (had to bring from ./../main.h here because feature of including headers in clBuildProgram does not work for some reason)
//========================================================================================================================================================================================================200

#define fp float

#define NUMBER_PAR_PER_BOX 100							// keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 128								// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))			// STABLE

//===============================================================================================================================================================================================================200
//	STRUCTURES (had to bring from ./../main.h here because feature of including headers in clBuildProgram does not work for some reason)
//===============================================================================================================================================================================================================200

typedef struct
{
	fp x, y, z;

} THREE_VECTOR;

typedef struct
{
	fp v, x, y, z;

} FOUR_VECTOR;

typedef struct nei_str
{

	// neighbor box
	int x, y, z;
	int number;
	long offset;

} nei_str;

typedef struct box_str
{

	// home box
	int x, y, z;
	int number;
	long offset;

	// neighbor boxes
	int nn;
	nei_str nei[26];

} box_str;

//========================================================================================================================================================================================================200
//	kernel_gpu_opencl KERNEL
//========================================================================================================================================================================================================200

__attribute__((max_global_work_dim(0)))
__kernel void kernel_gpu_opencl(fp                    alpha,
                                long                  number_boxes,
                       __global box_str*     restrict box,
                       __global FOUR_VECTOR* restrict rv,
                       __global fp*          restrict qv,
                       __global FOUR_VECTOR* restrict fv)
{
	fp a2=2.0*alpha*alpha;

	for(int l = 0; l < number_boxes; l++)
	{
		//------------------------------------------------------------------------------------------100
		//	home box - box parameters
		//------------------------------------------------------------------------------------------100

		long first_i = box[l].offset;										// offset to common arrays

		//------------------------------------------------------------------------------------------100
		//	Do for the # of (home+neighbor) boxes
		//------------------------------------------------------------------------------------------100

		for (int k = 0; k < (1+box[l].nn); k++) 
		{
			//----------------------------------------50
			//	neighbor box - get pointer to the right box
			//----------------------------------------50

			int pointer;

			if(k==0)
			{
				pointer = l;											// set first box to be processed to home box
			}
			else
			{
				pointer = box[l].nei[k-1].number;							// remaining boxes are neighbor boxes
			}

			//----------------------------------------50
			//	neighbor box - box parameters
			//----------------------------------------50

			long first_j = box[pointer].offset; 

			//----------------------------------------50
			//	Do for the # of particles in home box
			//----------------------------------------50

			for (int i = 0; i < NUMBER_PAR_PER_BOX; i++)
			{
				// shift registers for reduction
				fp v_SR[REDUCTION_LATENCY + 1], x_SR[REDUCTION_LATENCY + 1], y_SR[REDUCTION_LATENCY + 1], z_SR[REDUCTION_LATENCY + 1];

				// initialize shift registers
				#pragma unroll
				for (int j = 0; j < REDUCTION_LATENCY + 1; j++)
				{
					v_SR[j] = 0;
					x_SR[j] = 0;
					y_SR[j] = 0;
					z_SR[j] = 0;
				}

				// do for the # of particles in current (home or neighbor) box
				#pragma unroll UNROLL
				for (int j = 0; j < NUMBER_PAR_PER_BOX; j++)
				{
					THREE_VECTOR d;

					// coefficients
					fp r2 = rv[first_i+i].v + rv[first_j+j].v - DOT(rv[first_i+i],rv[first_j+j]);
					fp u2 = a2*r2;
					fp vij= exp(-u2);
					fp fs = 2.*vij;
					d.x = rv[first_i+i].x - rv[first_j+j].x;
					d.y = rv[first_i+i].y - rv[first_j+j].y;
					d.z = rv[first_i+i].z - rv[first_j+j].z;
					fp fxij=fs*d.x;
					fp fyij=fs*d.y;
					fp fzij=fs*d.z;

					// forces
					v_SR[REDUCTION_LATENCY] = v_SR[0] + qv[first_j+j]*vij;
					x_SR[REDUCTION_LATENCY] = x_SR[0] + qv[first_j+j]*fxij;
					y_SR[REDUCTION_LATENCY] = y_SR[0] + qv[first_j+j]*fyij;
					z_SR[REDUCTION_LATENCY] = z_SR[0] + qv[first_j+j]*fzij;

					// shift left
					#pragma unroll
					for (int m = 0; m < REDUCTION_LATENCY; m++)
					{
						v_SR[m] = v_SR[m + 1];
						x_SR[m] = x_SR[m + 1];
						y_SR[m] = y_SR[m + 1];
						z_SR[m] = z_SR[m + 1];
					}
				} // for j

				// final reduction
				#pragma unroll
				for (int j = 0; j < REDUCTION_LATENCY; j++)
				{
					fv[first_i+i].v  += v_SR[j];
					fv[first_i+i].x  += x_SR[j];
					fv[first_i+i].y  += y_SR[j];
					fv[first_i+i].z  += z_SR[j];
				}
			} // for i
		} // for k
	} // for l
}
