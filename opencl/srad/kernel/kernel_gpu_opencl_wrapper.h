#ifdef __cplusplus
extern "C" {
#endif
	
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER HEADER
//========================================================================================================================================================================================================200

void kernel_gpu_opencl_wrapper(	fp* image,					// input image
                                cl_int Nr,					// IMAGE nbr of rows
                                cl_int Nc,					// IMAGE nbr of cols
                                cl_long Ne,					// IMAGE nbr of elem
                                int niter,					// nbr of iterations
                                fp lambda,					// update step size
                                cl_long NeROI,					// ROI nbr of elements
                                int* iN,
                                int* iS,
                                int* jE,
                                int* jW,
                                int mem_size_i,
                                int mem_size_j,
                                int version,
//                                double* kernelRunTime,			// For calculating kernel execution time
                                double* extractTime,				// For image compression kernel (before compute loop)
                                double* computeTime,				// For the compute loop, similar to the CUDA version of the benchmark
                                double* compressTime				// For image compression kernel (after compute loop)
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
                             ,  double* power					// Power usage for supported boards
#endif
                             );

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
