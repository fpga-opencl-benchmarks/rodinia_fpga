//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./../main.h"								// (in the main program folder)

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>								// (in path known to compiler)	needed by printf
#include <string.h>								// (in path known to compiler)	needed by strlen

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./../util/opencl/opencl.h"				// (in directory)
#include "../../../common/timer.h"					// (in directory)
#include "../../common/opencl_util.h"				// (in directory)
#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
	#include "../../../common/power_fpga.h"
#endif

//======================================================================================================================================================150
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION HEADER
//======================================================================================================================================================150

#include "./kernel_gpu_opencl_wrapper.h"			// (in directory)

//======================================================================================================================================================150
//	END
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu_opencl_wrapper(	fp* image,					// input image
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
#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
                             ,  double* power					// Power usage for supported boards
#endif
                         )
{

	//======================================================================================================================================================150
	//	GPU SETUP
	//======================================================================================================================================================150

	//====================================================================================================100
	//	COMMON VARIABLES
	//====================================================================================================100

        int iter;
        // Originally passed as a parameter. Makes no sense.

	// common variables
	cl_int error;

#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
	// power measurement flag, only for Bittware's A10PL4 board
	int flag = 0;
#endif

	//====================================================================================================100
	//	GET PLATFORMS (Intel, AMD, NVIDIA, based on provided library), SELECT ONE
	//====================================================================================================100

	// Get the number of available platforms
	cl_uint num_platforms;
	char pbuf[100];
	
	// For run time calculation
	//cl_event (*kernelEvent)[4] = malloc(sizeof(cl_event[niter][4]));
	//cl_event kernelCompressEvent, kernelExtractEvent;
	//int i, j;
	//*kernelRunTime = 0;
	TimeStamp start[3], end[3];
#if 0        
	error = clGetPlatformIDs(	0, 
								NULL, 
								&num_platforms);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get the list of available platforms
	cl_platform_id *platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
	error = clGetPlatformIDs(	num_platforms, 
								platforms, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Select the 1st platform
	cl_platform_id platform = platforms[0];

	// Get the name of the selected platform and print it (if there are multiple platforms, choose the first one)

	error = clGetPlatformInfo(	platform, 
								CL_PLATFORM_VENDOR, 
								sizeof(pbuf), 
								pbuf, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	printf("Platform: %s\n", pbuf);

	// Create context properties for selected platform
	cl_context_properties context_properties[3] = {	CL_CONTEXT_PLATFORM, 
													(cl_context_properties) platform, 
													0};
        cl_device_type device_type = CL_DEVICE_TYPE_ALL;
#else
        cl_platform_id *platforms = NULL;
        cl_context_properties context_properties[3];
        cl_device_type device_type;
        display_device_info(&platforms, &num_platforms);
        select_device_type(platforms, &num_platforms, &device_type);
        validate_selection(platforms, &num_platforms, context_properties, &device_type);
#endif


	//====================================================================================================100
	//	CREATE CONTEXT FOR THE PLATFORM
	//====================================================================================================100

	// Create context for selected platform being GPU
	cl_context context;
	context = clCreateContextFromType(	context_properties, 
                                                device_type,
                                                NULL, 
                                                NULL, 
                                                &error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	GET DEVICES AVAILABLE FOR THE CONTEXT, SELECT ONE
	//====================================================================================================100

	// Get the number of devices (previousely selected for the context)
	size_t devices_size;
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								0, 
								NULL, 
								&devices_size);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Get the list of devices (previousely selected for the context)
	cl_device_id *devices = (cl_device_id *) malloc(devices_size);
	error = clGetContextInfo(	context, 
								CL_CONTEXT_DEVICES, 
								devices_size, 
								devices, 
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// Select the first device (previousely selected for the context) (if there are multiple devices, choose the first one)
	cl_device_id device;
	device = devices[0];

	// Get the name of the selected device (previousely selected for the context) and print it
	error = clGetDeviceInfo(device, 
							CL_DEVICE_NAME, 
							sizeof(pbuf), 
							pbuf, 
							NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	//printf("Device: %s\n", pbuf);

	//====================================================================================================100
	//	CREATE COMMAND QUEUE FOR THE DEVICE
	//====================================================================================================100

	// Create a command queue
	cl_command_queue command_queue;
	command_queue = clCreateCommandQueue(	context, 
											device, 
											CL_QUEUE_PROFILING_ENABLE, 
											&error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	CREATE PROGRAM, COMPILE IT
	//====================================================================================================100

	// Load kernel source code from file
	size_t sourceSize = 0;
	char *kernel_file_path = getVersionedKernelName("./kernel/srad_kernel", version);     
	char *source = read_kernel(kernel_file_path, &sourceSize);
	free(kernel_file_path);


#if defined(USE_JIT)        
	// Create the program
	cl_program program = clCreateProgramWithSource(	context, 
							1, 
							(const char **)&source, 
							&sourceSize, 
							&error);
#else
	cl_program program = clCreateProgramWithBinary(	context,
                                                        1,
                                                        devices,
                                                        &sourceSize,
                                                        (const unsigned char**)&source,
                                                        NULL,
                                                        &error);
#endif
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	
	free(source);

#if defined(USE_JIT)        
  char clOptions[1024];                                                  
  sprintf(clOptions,"-I. -DBSIZE=32 -DSSIZE=4");
#ifdef RD_WG_SIZE
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE=%d", RD_WG_SIZE);
#endif
#ifdef RD_WG_SIZE_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0=%d", RD_WG_SIZE_0);
#endif
#ifdef RD_WG_SIZE_0_0
  sprintf(clOptions + strlen(clOptions), " -DRD_WG_SIZE_0_0=%d", RD_WG_SIZE_0_0);
#endif

#ifdef FP_SINGLE
  sprintf(clOptions + strlen(clOptions), " -cl-single-precision-constant");
#endif

	//fprintf(stderr, "kernel compile options: %s\n", clOptions);
  
	// Compile the program
	clBuildProgram_SAFE(program, 1, &device, clOptions, NULL, NULL);
	
	/*// Print warnings and errors from compilation
	static char log[65536]; 
	memset(log, 0, sizeof(log));
	clGetProgramBuildInfo(	program, 
							device, 
							CL_PROGRAM_BUILD_LOG, 
							sizeof(log)-1, 
							log, 
							NULL);
	printf("-----OpenCL Compiler Output-----\n");
	if (strstr(log,"warning:") || strstr(log, "error:")) 
		printf("<<<<\n%s\n>>>>\n", log);
	printf("--------------------------------\n");
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);*/
#endif // USE_JIT

	//====================================================================================================100
	//	CREATE Kernels
	//====================================================================================================100

	cl_kernel extract_kernel = NULL, compute_kernel = NULL, prepare_kernel = NULL, reduce_kernel = NULL, srad_kernel = NULL, srad2_kernel = NULL, compress_kernel = NULL;

	// Extract kernel
	extract_kernel = clCreateKernel(program, "extract_kernel", &error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	if (!is_ndrange_kernel(version) && version > 3)
	{
		// Combined compute kernel
		compute_kernel = clCreateKernel(program, "compute_kernel", &error);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}
	else
	{
		// Prepare kernel
		prepare_kernel = clCreateKernel(program, "prepare_kernel", &error);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// Reduce kernel
		reduce_kernel = clCreateKernel(program, "reduce_kernel", &error);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// SRAD kernel
		srad_kernel = clCreateKernel(program, "srad_kernel", &error);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		// SRAD2 kernel
		srad2_kernel = clCreateKernel(program, "srad2_kernel", &error);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}

	// Compress kernel
	compress_kernel = clCreateKernel(program, "compress_kernel", &error);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//======================================================================================================================================================150
	// 	GPU VARIABLES
	//======================================================================================================================================================150

	// Kernel execution parameters
	int blocks_x;

	//======================================================================================================================================================150
	// 	ALLOCATE MEMORY IN GPU
	//======================================================================================================================================================150

	//====================================================================================================100
	// common memory size
	//====================================================================================================100

	int mem_size;														// matrix memory size
	mem_size = sizeof(fp) * Ne;												// get the size of float representation of input IMAGE

	//====================================================================================================100
	// allocate memory for entire IMAGE on DEVICE
	//====================================================================================================100

	cl_mem d_I;
	d_I = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// allocate memory for coordinates on DEVICE
	//====================================================================================================100

	cl_mem d_iN, d_iS, d_jW, d_jE;
	if (is_ndrange_kernel(version) || version < 5)
	{
		
		d_iN = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size_i,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		d_iS = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size_i,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		d_jE = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size_j,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		d_jW = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size_j,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}

	//====================================================================================================100
	// allocate memory for derivatives
	//====================================================================================================100

	cl_mem d_dN, d_dS, d_dW, d_dE;
	if (is_ndrange_kernel(version) || version < 5)
	{
		d_dN = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		d_dS = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		d_dW = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		d_dE = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}

	//====================================================================================================100
	// allocate memory for coefficient on DEVICE
	//====================================================================================================100

	cl_mem d_c;
	int d_c_mem_size = (is_ndrange_kernel(version) || version < 5) ? mem_size : sizeof(fp) * Ne;		// in fully-optimized single work-item kernels, d_c is only used to resolve bottom memory dependancy for block boundaries
	
	d_c = clCreateBuffer(	context, 
							CL_MEM_READ_WRITE, 
							d_c_mem_size,
							NULL, 
							&error );
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// allocate memory for partial sums on DEVICE
	//====================================================================================================100

	cl_mem d_sums, d_sums2;
	if (is_ndrange_kernel(version) || version < 5)								// no need for these if using combined kernel
	{
		d_sums = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		d_sums2 = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size, NULL, &error);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}
	
	//====================================================================================================100
	// allocate memory for output
	//====================================================================================================100

	cl_mem d_I_out;
	if (!is_ndrange_kernel(version) && version > 3)
	{
		d_I_out = clCreateBuffer(	context, 
								CL_MEM_READ_WRITE, 
								mem_size,
								NULL, 
								&error );
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}

	//====================================================================================================100
	// End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	COPY INPUT TO CPU
	//======================================================================================================================================================150

	//====================================================================================================100
	// Image
	//====================================================================================================100

	error = clEnqueueWriteBuffer(	command_queue, 
									d_I, 
									1, 
									0, 
									mem_size, 
									image, 
									0, 
									0, 
									0);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// coordinates
	//====================================================================================================100

	if (is_ndrange_kernel(version) || version < 5)
	{
		error = clEnqueueWriteBuffer(	command_queue, 
										d_iN, 
										1, 
										0, 
										mem_size_i, 
										iN, 
										0, 
										0, 
										0);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		error = clEnqueueWriteBuffer(	command_queue, 
										d_iS, 
										1, 
										0, 
										mem_size_i, 
										iS, 
										0, 
										0, 
										0);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		error = clEnqueueWriteBuffer(	command_queue, 
										d_jE, 
										1, 
										0, 
										mem_size_j, 
										jE, 
										0, 
										0, 
										0);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);

		error = clEnqueueWriteBuffer(	command_queue, 
										d_jW, 
										1, 
										0, 
										mem_size_j, 
										jW, 
										0, 
										0, 
										0);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}

	//====================================================================================================100
	// End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	KERNEL EXECUTION PARAMETERS
	//======================================================================================================================================================150

	// threads
	size_t local_work_size[1];
	local_work_size[0] = NUMBER_THREADS;

	// workgroups
	int blocks_work_size;
	size_t global_work_size[1];
	blocks_x = Ne/(int)local_work_size[0];
	if (Ne % (int)local_work_size[0] != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks_work_size = blocks_x;
	global_work_size[0] = blocks_work_size * local_work_size[0];						// define the number of blocks in the grid

	printf("max # of workgroups = %d, # of threads/workgroup = %d (ensure that device can handle)\n", (int)(global_work_size[0]/local_work_size[0]), (int)local_work_size[0]);

	//======================================================================================================================================================150
	// 	Extract Kernel - SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//======================================================================================================================================================150

	//====================================================================================================100
	//	set arguments
	//====================================================================================================100

	CL_SAFE_CALL( clSetKernelArg( extract_kernel, 0, sizeof(cl_long), (void *) &Ne) );
	CL_SAFE_CALL( clSetKernelArg( extract_kernel, 1, sizeof(cl_mem), (void *) &d_I) );

	//====================================================================================================100
	//	launch kernel
	//====================================================================================================100

	GetTime(start[0]);
        if (is_ndrange_kernel(version)) {
          error = clEnqueueNDRangeKernel(	command_queue, 
                                                extract_kernel, 
                                                1, 
                                                NULL, 
                                                global_work_size, 
                                                local_work_size, 
                                                0, 
                                                NULL, 
                                                NULL);
        } else {
          error = clEnqueueTask(command_queue, extract_kernel, 0, NULL, NULL);
        }
        
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	//	Synchronization - wait for all operations in the command queue so far to finish
	//====================================================================================================100

	 error = clFinish(command_queue);
	 if (error != CL_SUCCESS) 
		 fatal_CL(error, __LINE__);
	 GetTime(end[0]);

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	WHAT IS CONSTANT IN COMPUTATION LOOP
	//======================================================================================================================================================150

	cl_int blocks2_work_size;
	size_t global_work_size2[1];
	cl_long no;
	cl_int mul;
	int mem_size_single = sizeof(fp) * 1;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;
	int reduce_kernel_arg_idx = 0, srad_kernel_arg_idx = 0, srad2_kernel_arg_idx = 0;

	if (!is_ndrange_kernel(version) && version > 3)
	{
		CL_SAFE_CALL( clSetKernelArg(compute_kernel, 0, sizeof(fp),      (void *) &lambda)  );
		CL_SAFE_CALL( clSetKernelArg(compute_kernel, 1, sizeof(cl_int),  (void *) &Nr)      );
		CL_SAFE_CALL( clSetKernelArg(compute_kernel, 2, sizeof(cl_int),  (void *) &Nc)      );
		CL_SAFE_CALL( clSetKernelArg(compute_kernel, 3, sizeof(cl_mem),  (void *) &d_I)     );
		CL_SAFE_CALL( clSetKernelArg(compute_kernel, 4, sizeof(cl_mem),  (void *) &d_c)     );
		CL_SAFE_CALL( clSetKernelArg(compute_kernel, 5, sizeof(cl_mem),  (void *) &d_I_out) );
	}
	else
	{
		//====================================================================================================100
		//	Prepare Kernel
		//====================================================================================================100

		CL_SAFE_CALL( clSetKernelArg(prepare_kernel, 0, sizeof(cl_long), (void *) &Ne) );
		CL_SAFE_CALL( clSetKernelArg(prepare_kernel, 1, sizeof(cl_mem), (void *) &d_I) );
		CL_SAFE_CALL( clSetKernelArg(prepare_kernel, 2, sizeof(cl_mem), (void *) &d_sums) );
		CL_SAFE_CALL( clSetKernelArg(prepare_kernel, 3, sizeof(cl_mem), (void *) &d_sums2) );

		//====================================================================================================100
		//	Reduce Kernel
		//====================================================================================================100

		reduce_kernel_arg_idx = 0;
		CL_SAFE_CALL( clSetKernelArg( reduce_kernel, reduce_kernel_arg_idx, sizeof(cl_long), (void *) &Ne) );
		
		++reduce_kernel_arg_idx;
		
		if (is_ndrange_kernel(version)) {
		reduce_kernel_arg_idx += 2;
		}
		
		CL_SAFE_CALL( clSetKernelArg( reduce_kernel, reduce_kernel_arg_idx, sizeof(cl_mem), (void *) &d_sums) );
		
		++reduce_kernel_arg_idx;
		
		CL_SAFE_CALL( clSetKernelArg( reduce_kernel, reduce_kernel_arg_idx, sizeof(cl_mem), (void *) &d_sums2) );

		//====================================================================================================100
		//	SRAD Kernel
		//====================================================================================================100
		srad_kernel_arg_idx = 0;
		
		if (version == 0) {
			CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(fp), (void *) &lambda) );
		}
		
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_int), (void *) &Nr) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_int), (void *) &Nc) );

		if (is_ndrange_kernel(version)) {
			CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_long), (void *) &Ne) );
		}
		
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_iN) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_iS) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_jE) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_jW) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dN) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dS) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dW) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dE) );

		// skip for q0sqr
		++srad_kernel_arg_idx;
		
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_c) );
		CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_I) );

		//====================================================================================================100
		//	SRAD2 Kernel
		//====================================================================================================100
		srad2_kernel_arg_idx = 0;
		
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(fp), (void *) &lambda) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_int), (void *) &Nr) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_int), (void *) &Nc) );
		
		if (is_ndrange_kernel(version)) {
			CL_SAFE_CALL( clSetKernelArg(srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_long), (void *) &Ne) );
		}
		
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_iS) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_jE) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dN) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dS) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dW) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_dE) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_c) );
		CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_I) );
		if (!is_ndrange_kernel(version) && version != 1)
		{
			CL_SAFE_CALL( clSetKernelArg( srad2_kernel, srad2_kernel_arg_idx++, sizeof(cl_mem), (void *) &d_I) );
		}
	}

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	COMPUTATION
	//======================================================================================================================================================150

	if (version < 5) //disable iteration progress for single-kernel versions
	{
		printf("\n==============================================================\n");
		printf("Iterations Progress: ");
	}

#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
	#pragma omp parallel num_threads(2) shared(flag)
	{
		if (omp_get_thread_num() == 0)
		{
			*power = GetPowerFPGA(&flag);
		}
		else
		{
			#pragma omp barrier
#endif
			// start of timing point
			GetTime(start[1]);

			// execute main loop
			for (iter=0; iter<niter; iter++) // do for the number of iterations input parameter
			{
				// to avoid too much noise in the output, print iteration number once every 10 iterations, disabled for single-kernel versions
				if (iter % 10 == 0 && version < 5)
				{
					printf("%d ", iter);
				}
				fflush(NULL);

				//====================================================================================================100
				// Combined compute kernel for FPGA
				//====================================================================================================100

				if (!is_ndrange_kernel(version) && version > 3)
				{
					error = clEnqueueTask(command_queue, compute_kernel, 0, NULL, NULL);   
					if (error != CL_SUCCESS) 
						fatal_CL(error, __LINE__);

					// swapping input and output image buffers
					if (iter%2 == 0)
					{
						CL_SAFE_CALL( clSetKernelArg(compute_kernel, 3, sizeof(cl_mem),  (void *) &d_I_out) );
						CL_SAFE_CALL( clSetKernelArg(compute_kernel, 5, sizeof(cl_mem),  (void *) &d_I)     );
					}
					else
					{
						CL_SAFE_CALL( clSetKernelArg(compute_kernel, 3, sizeof(cl_mem),  (void *) &d_I)     );
						CL_SAFE_CALL( clSetKernelArg(compute_kernel, 5, sizeof(cl_mem),  (void *) &d_I_out) );
					}
				}
				else
				{
					//====================================================================================================100
					// Prepare kernel
					//====================================================================================================100

					// launch kernel
					if (is_ndrange_kernel(version)) {
					error = clEnqueueNDRangeKernel(	command_queue, 
													prepare_kernel, 
													1, 
													NULL, 
													global_work_size, 
													local_work_size, 
													0, 
													NULL, 
													NULL);
					} else {
					error = clEnqueueTask(command_queue, prepare_kernel, 0, NULL, NULL);
					}
					
					if (error != CL_SUCCESS) 
						fatal_CL(error, __LINE__);

					// synchronize
					// error = clFinish(command_queue);
					// if (error != CL_SUCCESS) 
						// fatal_CL(error, __LINE__);

					//====================================================================================================100
					//	Reduce Kernel - performs subsequent reductions of sums
					//====================================================================================================100

					if (is_ndrange_kernel(version)) {
					// initial values
					blocks2_work_size = blocks_work_size;							// original number of blocks
					global_work_size2[0] = global_work_size[0];
					no = Ne;										// original number of sum elements
					mul = 1;										// original multiplier

					// loop
					while(blocks2_work_size != 0){

					// set arguments that were uptaded in this loop
					CL_SAFE_CALL( clSetKernelArg( reduce_kernel, 1, sizeof(cl_long), (void *) &no) );
					CL_SAFE_CALL( clSetKernelArg( reduce_kernel, 2, sizeof(cl_int), (void *) &mul) );
					CL_SAFE_CALL( clSetKernelArg( reduce_kernel, 5, sizeof(cl_int), (void *) &blocks2_work_size) );

					// launch kernel
					error = clEnqueueNDRangeKernel(	command_queue, 
										reduce_kernel, 
										1, 
										NULL, 
										global_work_size2, 
										local_work_size, 
										0, 
										NULL, 
										NULL);
					if (error != CL_SUCCESS) 
						fatal_CL(error, __LINE__);

					// synchronize
					// error = clFinish(command_queue);
					// if (error != CL_SUCCESS) 
					// fatal_CL(error, __LINE__);

					// update execution parameters
					no = blocks2_work_size;							// get current number of elements
					if(blocks2_work_size == 1){
					blocks2_work_size = 0;
					}
					else{
					mul = mul * NUMBER_THREADS;						// update the increment
					blocks_x = blocks2_work_size/(int)local_work_size[0];			// number of blocks
					if (blocks2_work_size % (int)local_work_size[0] != 0){			// compensate for division remainder above by adding one grid
						blocks_x = blocks_x + 1;
					}
					blocks2_work_size = blocks_x;
					global_work_size2[0] = blocks2_work_size * (int)local_work_size[0];
					}
						
					}
					} else {
					error = clEnqueueTask(command_queue, reduce_kernel, 0, NULL, NULL);
					}

					// copy total sums to device
					error = clEnqueueReadBuffer(command_queue,
												d_sums,
												CL_TRUE,
												0,
												mem_size_single,
												&total,
												0,
												NULL,
												NULL);
					if (error != CL_SUCCESS) 
						fatal_CL(error, __LINE__);

					error = clEnqueueReadBuffer(command_queue,
												d_sums2,
												CL_TRUE,
												0,
												mem_size_single,
												&total2,
												0,
												NULL,
												NULL);
					if (error != CL_SUCCESS) 
						fatal_CL(error, __LINE__);

					//====================================================================================================100
					// calculate statistics
					//====================================================================================================100
					
					meanROI  = total / (fp)(NeROI);										// gets mean (average) value of element in ROI
					meanROI2 = meanROI * meanROI;										//
					varROI   = (total2 / (fp)(NeROI)) - meanROI2;								// gets variance of ROI
					q0sqr    = varROI / meanROI2;										// gets standard deviation of ROI

					//====================================================================================================100
					// execute srad kernel
					//====================================================================================================100

					// set arguments that were uptaded in this loop
					CL_SAFE_CALL( clSetKernelArg( srad_kernel, srad_kernel_arg_idx - 3, sizeof(fp), (void *) &q0sqr) );

					// launch kernel
					if (is_ndrange_kernel(version)) {
					error = clEnqueueNDRangeKernel(	command_queue, 
										srad_kernel, 
										1, 
										NULL, 
										global_work_size, 
										local_work_size, 
										0, 
										NULL, 
										NULL);
					} else {
					error = clEnqueueTask(command_queue, srad_kernel, 0, NULL, NULL);
					}
					
					if (error != CL_SUCCESS) 
						fatal_CL(error, __LINE__);

					// synchronize
					// error = clFinish(command_queue);
					// if (error != CL_SUCCESS) 
						// fatal_CL(error, __LINE__);

					//====================================================================================================100
					// execute srad2 kernel
					//====================================================================================================100

					// launch kernel
					if (is_ndrange_kernel(version)) {
					error = clEnqueueNDRangeKernel(	command_queue, 
										srad2_kernel, 
										1, 
										NULL, 
										global_work_size, 
										local_work_size, 
										0, 
										NULL, 
										NULL);
					} else {
					error = clEnqueueTask(command_queue, srad2_kernel, 0, NULL, NULL);
					}
					
					if (error != CL_SUCCESS) 
						fatal_CL(error, __LINE__);

					//====================================================================================================100
					// End
					//====================================================================================================100
				}
			}

			//====================================================================================================100
			// synchronize
			//====================================================================================================100

			error = clFinish(command_queue);
			if (error != CL_SUCCESS) 
				fatal_CL(error, __LINE__);
			GetTime(end[1]);
#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
			flag = 1;
		}
	}
#endif

	if (version < 5) // disable iteration progress for single-kernel versions
	{
		printf("\n==============================================================\n\n");
	}

	//======================================================================================================================================================150
	// 	Compress Kernel - SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//======================================================================================================================================================150

	//====================================================================================================100
	// set parameters
	//====================================================================================================100

	CL_SAFE_CALL( clSetKernelArg( compress_kernel, 0, sizeof(cl_long), (void *) &Ne) );
	cl_mem output_source = (is_ndrange_kernel(version) || version < 5) ? d_I : (niter%2 == 0) ? d_I : d_I_out;
	CL_SAFE_CALL( clSetKernelArg( compress_kernel, 1, sizeof(cl_mem), (void *) &output_source) );

	//====================================================================================================100
	// launch kernel
	//====================================================================================================100

	GetTime(start[2]);
        if (is_ndrange_kernel(version)) {
            error = clEnqueueNDRangeKernel(	command_queue, 
                                                compress_kernel, 
                                                1, 
                                                NULL, 
                                                global_work_size, 
                                                local_work_size, 
                                                0, 
                                                NULL, 
                                                NULL);
            
        } else {
            error = clEnqueueTask(	command_queue, 
                                        compress_kernel, 
                                        0, 
                                        NULL, 
                                        NULL);
        }
          
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	//====================================================================================================100
	// synchronize
	//====================================================================================================100

	error = clFinish(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	GetTime(end[2]);

	//====================================================================================================100
	//	End
	//====================================================================================================100

	//======================================================================================================================================================150
	// 	COPY RESULTS BACK TO CPU
	//======================================================================================================================================================150

	error = clEnqueueReadBuffer(command_queue,
								output_source,
								CL_TRUE,
								0,
								mem_size,
								image,
								0,
								NULL,
								NULL);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// int i;
	// for(i=0; i<100; i++){
		// printf("%f ", image[i]);
	// }

	//======================================================================================================================================================150
	// 	FREE MEMORY
	//======================================================================================================================================================150

	// OpenCL structures
	error = clReleaseKernel(extract_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	if(!is_ndrange_kernel(version) && version > 3)
	{
		error = clReleaseKernel(compute_kernel);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}
	else
	{
		error = clReleaseKernel(prepare_kernel);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseKernel(reduce_kernel);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseKernel(srad_kernel);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseKernel(srad2_kernel);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}
	error = clReleaseKernel(compress_kernel);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseProgram(program);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	// common_change
	error = clReleaseMemObject(d_I);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	if(!is_ndrange_kernel(version) && version > 3)
	{
		error = clReleaseMemObject(d_I_out);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}
	error = clReleaseMemObject(d_c);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);

	if(is_ndrange_kernel(version) || version < 5)
	{
		error = clReleaseMemObject(d_iN);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_iS);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_jE);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_jW);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_sums);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_sums2);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_dN);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_dS);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_dE);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
		error = clReleaseMemObject(d_dW);
		if (error != CL_SUCCESS) 
			fatal_CL(error, __LINE__);
	}

	// OpenCL structures
	error = clFlush(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseCommandQueue(command_queue);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	error = clReleaseContext(context);
	if (error != CL_SUCCESS) 
		fatal_CL(error, __LINE__);
	
	// Kernel run time calculation
	/*for (i = 0; i < niter; i++)
	{
		for (j = 0; j < 4; j++)
		{
			*kernelRunTime += CLTimeDiff( CLGetTime(kernelEvent[i][j], START), CLGetTime(kernelEvent[i][j], END) );
		}
	}*/
	//*compressTime = CLTimeDiff( CLGetTime(kernelCompressEvent, START), CLGetTime(kernelCompressEvent, END) );
	*compressTime = TimeDiff(start[0], end[0]);
	*computeTime = TimeDiff(start[1], end[1]);
	*extractTime = TimeDiff(start[2], end[2]);
	//*extractTime = CLTimeDiff( CLGetTime(kernelExtractEvent, START), CLGetTime(kernelExtractEvent, END) );
	//*kernelRunTime = *kernelRunTime + *compressTime + *extractTime;
	//free(kernelEvent);

	//======================================================================================================================================================150
	// 	End
	//======================================================================================================================================================150

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
