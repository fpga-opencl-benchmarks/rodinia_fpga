/*
 * =====================================================================================
 *
 *       Filename:  lud.cu
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>
#ifdef AOCL_BOARD_a10pl4_gx115es3
	#include "../../../common/power_fpga.h"
#endif

#include "common.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
using std::string;

#include "../../common/opencl_util.h"
#include "../../../common/timer.h"
#ifndef BLOCK_SIZE
	#include "problem_size.h"
#endif

static cl_context       context;
static cl_command_queue cmd_queue;
static cl_command_queue cmd_queue2;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

static int initialize()
{
	size_t size;
        cl_int err;
	// create OpenCL context
#if 0
	cl_platform_id platform_id;
	if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(1,*,0) failed\n"); return -1; }
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
#else
        cl_platform_id *platforms = NULL;
        cl_uint num_platforms = 0;
        cl_context_properties ctxprop[3];
        display_device_info(&platforms, &num_platforms);
        select_device_type(platforms, &num_platforms, &device_type);
        validate_selection(platforms, &num_platforms, ctxprop, &device_type);
#endif

	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, &err );
	if ( err != CL_SUCCESS ) {
          display_error_message(err, stderr);
          return -1;
        }

	// get the list of GPUs
	CL_SAFE_CALL(clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size ));
	num_devices = (int) (size / sizeof(cl_device_id));
	//printf("num_devices = %d\n", num_devices);

	if( num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	CL_SAFE_CALL(clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL ));

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
	cmd_queue2 = clCreateCommandQueue( context, device_list[0], 0, NULL );
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
	if( !cmd_queue2 ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( cmd_queue2 ) clReleaseCommandQueue( cmd_queue2 );
	if( context ) clReleaseContext( context );
	if( device_list ) delete device_list;

	// reset all variables
	cmd_queue = 0;
	cmd_queue2 = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

static int do_verify = 0;
void lud_cuda(float *d_m, int matrix_dim);

static struct option long_options[] = {
      /* name, has_arg, flag, val */
      {"input", 1, NULL, 'i'},
      {"size", 1, NULL, 's'},
      {"verify", 0, NULL, 'v'},
      {0,0,0,0}
};

int
main ( int argc, char *argv[] )
{
	printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
	int matrix_dim = 32; /* default matrix_dim */
	int opt, option_index=0;
	func_ret_t ret;
	const char *input_file = NULL;
	float *m, *mm;
	int i, version;
	TimeStamp start, end;
	double totalTime;

#ifdef PROFILE
	TimeStamp start1, end1;
#endif
	size_t globalwork2, globalwork3;

#ifdef AOCL_BOARD_a10pl4_gx115es3
	// power measurement flags, only for Arria 10
	int flag = 0;
	double power = 0;
	double energy = 0;
#endif

	cl_kernel diagonal = NULL, perimeter = NULL, internal = NULL;	// for NDRange kernels
	cl_kernel perimeter_row = NULL, perimeter_col = NULL;		// for v8
	cl_kernel lud = NULL;						// for single work-item kernels

	cl_event dia_exec, peri_col_exec;				// events for v8

	// get kernel version from commandline
	init_fpga(&argc, &argv, &version);

        // Does Windows have getopt_long? This is just simple argument
        // handling, so if it's not available on Windows, just not use
        // the function.
	while ((opt = getopt_long(argc, argv, "::vs:i:", 
                                  long_options, &option_index)) != -1 ) {
          switch(opt){
            case 'i':
              input_file = optarg;
              break;
            case 'v':
              do_verify = 1;
              break;
            case 's':
              matrix_dim = atoi(optarg);
              printf("Generate input matrix internally, size =%d\n", matrix_dim);
              // fprintf(stderr, "Currently not supported, use -i instead\n");
              // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
              // exit(EXIT_FAILURE);
              break;
            case '?':
              fprintf(stderr, "invalid option\n");
              break;
            case ':':
              fprintf(stderr, "missing argument\n");
              break;
            default:
              fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n",
                      argv[0]);
              exit(EXIT_FAILURE);
          }
	}

	if ( (optind < argc) || (optind == 1)) {
		fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	if (input_file) {
		printf("Reading matrix from file %s\n", input_file);
		ret = create_matrix_from_file(&m, input_file, &matrix_dim);
		if (ret != RET_SUCCESS) {
			m = NULL;
			fprintf(stderr, "error create matrix from file %s\n", input_file);
			exit(EXIT_FAILURE);
		}
	}

	else if (matrix_dim) {
	  printf("Creating matrix internally size=%d\n", matrix_dim);
	  ret = create_matrix(&m, matrix_dim);
	  if (ret != RET_SUCCESS) {
	    m = NULL;
	    fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
	    exit(EXIT_FAILURE);
	  }
	}

	else {
	  printf("No input file specified!\n");
	  exit(EXIT_FAILURE);
	}

	if (do_verify){
		//printf("Before LUD\n");
		// print_matrix(m, matrix_dim);
		matrix_duplicate(m, &mm, matrix_dim);
	}
	
	//int sourcesize = 1024*1024;
	//char * source = (char *)calloc(sourcesize, sizeof(char)); 
	//if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }
	
	size_t sourcesize;
	char *kernel_file_path = getVersionedKernelName("./ocl/lud_kernel", version);
	char *source = read_kernel(kernel_file_path, &sourcesize);
	free(kernel_file_path);

	// OpenCL initialization
	if (initialize()) return -1;
	
	// compile kernel
	cl_int err = 0;
	
#if defined(USE_JIT)
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
#else
	cl_program prog = clCreateProgramWithBinary(context,
                                                    1,
                                                    device_list,
                                                    &sourcesize,
                                                    (const unsigned char**)&source,
                                                    NULL,
                                                    &err);
#endif

	if(err != CL_SUCCESS) {
          display_error_message(err, stderr);
          return -1;
        }
#if defined(USE_JIT)
	char clOptions[110];
	sprintf(clOptions, "-I.");
#ifdef BLOCK_SIZE
	sprintf(clOptions + strlen(clOptions), " -DBLOCK_SIZE=%d", BLOCK_SIZE);
#endif
	clBuildProgram_SAFE(prog, num_devices, device_list, clOptions, NULL, NULL);
#endif // USE_JIT        

	cl_mem d_m;
	d_m = clCreateBuffer(context, CL_MEM_READ_WRITE, matrix_dim*matrix_dim * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer d_m (size:%d) => %d\n", matrix_dim*matrix_dim, err); return -1;} 

	CL_SAFE_CALL(clEnqueueWriteBuffer(cmd_queue, d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0, 0, 0));

	if (version == 5)						// single-kernel version
	{
		lud = clCreateKernel(prog, "lud", &err);
		if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
		clReleaseProgram(prog);

		CL_SAFE_CALL( clSetKernelArg(lud, 0, sizeof(void *), (void*) &d_m       ) );
		CL_SAFE_CALL( clSetKernelArg(lud, 1, sizeof(cl_int), (void*) &matrix_dim) );

#ifdef AOCL_BOARD_a10pl4_gx115es3
		#pragma omp parallel num_threads(2) shared(flag)
		{
			if (omp_get_thread_num() == 0)
			{
				power = GetPowerFPGA(&flag);
			}
			else
			{
				#pragma omp barrier
#endif
				// beginning of timing point
				GetTime(start);

				CL_SAFE_CALL( clEnqueueTask(cmd_queue, lud, 0, NULL, NULL) );
				clFinish(cmd_queue);

				// end of timing point
				GetTime(end);
#ifdef AOCL_BOARD_a10pl4_gx115es3
				flag = 1;
			}
		}
#endif
	}
	else
	{
		diagonal  = clCreateKernel(prog, "lud_diagonal", &err);
		if (version == 8)
		{
			perimeter_row = clCreateKernel(prog, "lud_perimeter_row", &err);
			perimeter_col = clCreateKernel(prog, "lud_perimeter_col", &err);
		}
		else
		{
			perimeter = clCreateKernel(prog, "lud_perimeter", &err);
		}
		internal  = clCreateKernel(prog, "lud_internal", &err);  
		if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
		clReleaseProgram(prog); 

		if (version >= 4)
		{
			// fixed kernel arguments
			CL_SAFE_CALL( clSetKernelArg(diagonal,  0, sizeof(void *),                      (void*) &d_m       ) );
			CL_SAFE_CALL( clSetKernelArg(diagonal,  1, sizeof(cl_int),                      (void*) &matrix_dim) );
			
			if (version == 8)
			{
				CL_SAFE_CALL( clSetKernelArg(perimeter_row, 0, sizeof(void *),          (void*) &d_m       ) );
				CL_SAFE_CALL( clSetKernelArg(perimeter_row, 1, sizeof(cl_int),          (void*) &matrix_dim) );
			
				CL_SAFE_CALL( clSetKernelArg(perimeter_col, 0, sizeof(void *),          (void*) &d_m       ) );
				CL_SAFE_CALL( clSetKernelArg(perimeter_col, 1, sizeof(cl_int),          (void*) &matrix_dim) );
			}
			else
			{
				CL_SAFE_CALL( clSetKernelArg(perimeter, 0, sizeof(void *),              (void*) &d_m       ) );
				CL_SAFE_CALL( clSetKernelArg(perimeter, 1, sizeof(cl_int),              (void*) &matrix_dim) );
			}
			
			CL_SAFE_CALL( clSetKernelArg(internal,  0, sizeof(void *),                      (void*) &d_m       ) );
			CL_SAFE_CALL( clSetKernelArg(internal,  1, sizeof(cl_int),                      (void*) &matrix_dim) );
		}
		else
		{
			// fixed kernel arguments
			CL_SAFE_CALL( clSetKernelArg(diagonal,  0, sizeof(void *),                      (void*) &d_m       ) );
			CL_SAFE_CALL( clSetKernelArg(diagonal,  1, sizeof(float)*BLOCK_SIZE*BLOCK_SIZE, (void*) NULL       ) );
			CL_SAFE_CALL( clSetKernelArg(diagonal,  2, sizeof(cl_int),                      (void*) &matrix_dim) );
			
			CL_SAFE_CALL( clSetKernelArg(perimeter, 0, sizeof(void *),                      (void*) &d_m       ) );
			CL_SAFE_CALL( clSetKernelArg(perimeter, 1, sizeof(float)*BLOCK_SIZE*BLOCK_SIZE, (void*) NULL       ) );
			CL_SAFE_CALL( clSetKernelArg(perimeter, 2, sizeof(float)*BLOCK_SIZE*BLOCK_SIZE, (void*) NULL       ) );
			CL_SAFE_CALL( clSetKernelArg(perimeter, 3, sizeof(float)*BLOCK_SIZE*BLOCK_SIZE, (void*) NULL       ) );
			CL_SAFE_CALL( clSetKernelArg(perimeter, 4, sizeof(cl_int),                      (void*) &matrix_dim) );
			
			CL_SAFE_CALL( clSetKernelArg(internal,  0, sizeof(void *),                      (void*) &d_m       ) );
			CL_SAFE_CALL( clSetKernelArg(internal,  1, sizeof(float)*BLOCK_SIZE*BLOCK_SIZE, (void*) NULL       ) );
			CL_SAFE_CALL( clSetKernelArg(internal,  2, sizeof(float)*BLOCK_SIZE*BLOCK_SIZE, (void*) NULL       ) );
			CL_SAFE_CALL( clSetKernelArg(internal,  3, sizeof(cl_int),                      (void*) &matrix_dim) );
		}
		
		// fixed work sizes
		size_t global_work1[3] = {BLOCK_SIZE, 1, 1};
		size_t local_work1[3]  = {BLOCK_SIZE, 1, 1};

		size_t peri_work_size  = (version == 8) ? BLOCK_SIZE : BLOCK_SIZE * 2;
		size_t local_work2[3]  = {peri_work_size, 1, 1};

		size_t local_work3[3]  = {BLOCK_SIZE, BLOCK_SIZE, 1};

#ifdef AOCL_BOARD_a10pl4_gx115es3
		#pragma omp parallel num_threads(2) shared(flag)
		{
			if (omp_get_thread_num() == 0)
			{
				power = GetPowerFPGA(&flag);
			}
			else
			{
				#pragma omp barrier
#endif
				// beginning of timing point
				GetTime(start);
				for (i = 0; i < matrix_dim - BLOCK_SIZE; i += BLOCK_SIZE)
				{
					if (version >= 4)
					{
						CL_SAFE_CALL( clSetKernelArg(diagonal , 2, sizeof(cl_int), (void*) &i) );
						if (version == 8)
						{
							CL_SAFE_CALL( clSetKernelArg(perimeter_row, 2, sizeof(cl_int), (void*) &i) );
							CL_SAFE_CALL( clSetKernelArg(perimeter_col, 2, sizeof(cl_int), (void*) &i) );
						}
						else
						{
							CL_SAFE_CALL( clSetKernelArg(perimeter, 2, sizeof(cl_int), (void*) &i) );
						}
						CL_SAFE_CALL( clSetKernelArg(internal , 2, sizeof(cl_int), (void*) &i) );
					}
					else
					{
						CL_SAFE_CALL( clSetKernelArg(diagonal , 3, sizeof(cl_int), (void*) &i) );
						CL_SAFE_CALL( clSetKernelArg(perimeter, 5, sizeof(cl_int), (void*) &i) );
						CL_SAFE_CALL( clSetKernelArg(internal , 4, sizeof(cl_int), (void*) &i) );
					}
					
					globalwork2 = peri_work_size * (((matrix_dim-i)/BLOCK_SIZE) - 1);
					globalwork3 = BLOCK_SIZE * (((matrix_dim-i)/BLOCK_SIZE) - 1);
					
					size_t global_work2[3] = {globalwork2, 1, 1};
					size_t global_work3[3] = {globalwork3, globalwork3, 1};

					if (is_ndrange_kernel(version))
					{
						#ifdef PROFILE
						GetTime(start1);
						#endif
						
						CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, diagonal , 2, NULL, global_work1, local_work1, 0, 0, &dia_exec) );
						
						#ifdef PROFILE
						clFinish(cmd_queue);
						GetTime(end1);
						printf("%d: diameter: %f\n", i, TimeDiff(start1, end1));
						#endif

						#ifdef PROFILE
						GetTime(start1);
						#endif
						
						if (version == 8)
						{
							CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, perimeter_row, 2, NULL, global_work2, local_work2, 0, 0, 0) );
							//clFinish(cmd_queue);
							//GetTime(end1);
							//printf("%d: peri_row: %f\n", i, TimeDiff(start1, end1));
							//GetTime(start1);
							CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue2, perimeter_col, 2, NULL, global_work2, local_work2, 1, &dia_exec, &peri_col_exec) );
						}
						else
						{
							CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, perimeter, 2, NULL, global_work2, local_work2, 0, 0, 0) );
						}
						
						#ifdef PROFILE
						clFinish(cmd_queue);
						clFinish(cmd_queue2);
						GetTime(end1);
						printf("%d: perimete: %f\n", i, TimeDiff(start1, end1));
						#endif

						#ifdef PROFILE
						GetTime(start1);
						#endif

						if (version == 8)
						{
							CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, internal , 2, NULL, global_work3, local_work3, 1, &peri_col_exec, 0) );
						}
						else
						{
							CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, internal , 2, NULL, global_work3, local_work3, 0, 0, 0) );
						}
						
						#ifdef PROFILE
						clFinish(cmd_queue);
						GetTime(end1);
						printf("%d: internal: %f\n\n", i, TimeDiff(start1, end1));
						#endif
					}
					else
					{
						#ifdef PROFILE
						GetTime(start1);
						#endif
						
						CL_SAFE_CALL( clEnqueueTask(cmd_queue, diagonal , 0, NULL, NULL) );
						
						#ifdef PROFILE
						clFinish(cmd_queue);
						GetTime(end1);
						printf("%d: diameter: %f\n", i, TimeDiff(start1, end1));
						#endif

						#ifdef PROFILE
						GetTime(start1);
						#endif
						
						CL_SAFE_CALL( clEnqueueTask(cmd_queue, perimeter, 0, NULL, NULL) );
						
						#ifdef PROFILE
						clFinish(cmd_queue);
						GetTime(end1);
						printf("%d: perimete: %f\n", i, TimeDiff(start1, end1));
						#endif

						#ifdef PROFILE
						GetTime(start1);
						#endif
						
						CL_SAFE_CALL( clEnqueueTask(cmd_queue, internal , 0, NULL, NULL) );
						
						#ifdef PROFILE
						clFinish(cmd_queue);
						GetTime(end1);
						printf("%d: internal: %f\n\n", i, TimeDiff(start1, end1));
						#endif
					}
				}

				if (version >= 4)
				{
					CL_SAFE_CALL( clSetKernelArg(diagonal, 2, sizeof(cl_int), (void*) &i) );
				}
				else
				{
					CL_SAFE_CALL( clSetKernelArg(diagonal, 3, sizeof(cl_int), (void*) &i) );
				}

				if (is_ndrange_kernel(version))
				{
					CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, diagonal, 2, NULL, global_work1, local_work1, 0, 0, 0) );
				}
				else
				{
					CL_SAFE_CALL( clEnqueueTask(cmd_queue, diagonal, 0, NULL, NULL) );
				}

				clFinish(cmd_queue);
				// end of timing point
				GetTime(end);
#ifdef AOCL_BOARD_a10pl4_gx115es3
				flag = 1;
			}
		}
#endif
	}

	CL_SAFE_CALL( clEnqueueReadBuffer(cmd_queue, d_m, 1, 0, matrix_dim*matrix_dim*sizeof(float), m, 0, 0, 0) );
	clFinish(cmd_queue);
	clReleaseMemObject(d_m);

	totalTime = TimeDiff(start, end);
	printf("Computation done in %0.3lf ms.\n", totalTime);

#ifdef AOCL_BOARD_a10pl4_gx115es3
	energy = GetEnergyFPGA(power, totalTime);
	if (power != -1) // -1 --> failed to read energy values
	{
		printf("Total energy used is %0.3lf jouls.\n", energy);
		printf("Average power consumption is %0.3lf watts.\n", power);
	}
#endif

	if (do_verify){
		//printf("After LUD\n");
		// print_matrix(m, matrix_dim);
		printf("Verifying output: ");
		if (lud_verify(mm, m, matrix_dim) == RET_SUCCESS)
		{
			printf("verification succeeded!\n");
		}
		else
		{
			printf("verification failed!\n");
		}
		free(mm);
	}

	free(m);
	free(source);
	
	if(shutdown()) return -1;
	
}				

/* ----------  end of function main  ---------- */


