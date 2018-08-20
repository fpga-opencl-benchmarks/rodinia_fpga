//========================================================================================================================================================================================================200
//	INCLUDE/DEFINE (had to bring from ./../main.h here because feature of including headers in clBuildProgram does not work for some reason)
//========================================================================================================================================================================================================200

#define fp float

#define NUMBER_PAR_PER_BOX 100							// keep this low to allow more blocks that share shared memory to run concurrently, code does not work for larger than 110, more speedup can be achieved with larger number and no shared memory used

#define NUMBER_THREADS 128								// this should be roughly equal to NUMBER_PAR_PER_BOX for best performance

#define DOT(A,B) ((A.x)*(B.x)+(A.y)*(B.y)+(A.z)*(B.z))	// STABLE

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

__kernel void kernel_gpu_opencl(fp                    alpha,
                                long                  number_boxes,
                       __global box_str*     restrict box,
                       __global FOUR_VECTOR* restrict rv,
                       __global fp*          restrict qv,
                       __global FOUR_VECTOR* restrict fv)
{
	fp a2=2.0*alpha*alpha;
	__global FOUR_VECTOR* rA;
	__global FOUR_VECTOR* fA;
	__global FOUR_VECTOR* rB;
	__global fp* qB;

	for(int l=0; l<number_boxes; l=l+1){

		//------------------------------------------------------------------------------------------100
		//	home box - box parameters
		//------------------------------------------------------------------------------------------100

		long first_i = box[l].offset;												// offset to common arrays

		//------------------------------------------------------------------------------------------100
		//	home box - distance, force, charge and type parameters from common arrays
		//------------------------------------------------------------------------------------------100

		rA = &rv[first_i];
		fA = &fv[first_i];

		//------------------------------------------------------------------------------------------100
		//	Do for the # of (home+neighbor) boxes
		//------------------------------------------------------------------------------------------100

		for (int k=0; k<(1+box[l].nn); k++) 
		{

			//----------------------------------------50
			//	neighbor box - get pointer to the right box
			//----------------------------------------50

			int pointer;

			if(k==0){
				pointer = l;													// set first box to be processed to home box
			}
			else{
				pointer = box[l].nei[k-1].number;							// remaining boxes are neighbor boxes
			}

			//----------------------------------------50
			//	neighbor box - box parameters
			//----------------------------------------50

			long first_j = box[pointer].offset;

			//----------------------------------------50
			//	neighbor box - distance, force, charge and type parameters
			//----------------------------------------50

			rB = &rv[first_j];
			qB = &qv[first_j];

			//----------------------------------------50
			//	Do for the # of particles in home box
			//----------------------------------------50

			for (int i=0; i<NUMBER_PAR_PER_BOX; i=i+1){

				// do for the # of particles in current (home or neighbor) box
				for (int j=0; j<NUMBER_PAR_PER_BOX; j=j+1){
					THREE_VECTOR d;

					// coefficients
					fp r2  = rA[i].v + rB[j].v - DOT(rA[i],rB[j]); 
					fp u2  = a2*r2;
					fp vij = exp(-u2);
					fp fs  = 2.*vij;
					d.x = rA[i].x - rB[j].x; 
					d.y = rA[i].y - rB[j].y; 
					d.z = rA[i].z - rB[j].z;
					fp fxij = fs*d.x;
					fp fyij = fs*d.y;
					fp fzij = fs*d.z;

					// forces
					fA[i].v +=  qB[j]*vij;
					fA[i].x +=  qB[j]*fxij;
					fA[i].y +=  qB[j]*fyij;
					fA[i].z +=  qB[j]*fzij;

				} // for j

			} // for i

		} // for k

	} // for l

}
