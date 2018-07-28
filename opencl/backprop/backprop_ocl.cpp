// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backprop.h"

#ifdef NV //NVIDIA
	#include <oclUtils.h>
#else 
	#include <CL/cl.h>
#endif

#include "../common/opencl_util.h"
#include "../../common/timer.h"

////////////////////////////////////////////////////////////////////////////////

// global variables
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_device_type   device_type;
static cl_device_id*    device_list;
static cl_int           num_devices;
int version;

static int initialize()
{
	cl_int result;
	size_t size;

	// create OpenCL context
	cl_uint platformCount;
	cl_platform_id* platforms = NULL;
	cl_context_properties ctxprop[3];

	display_device_info(&platforms, &platformCount);
	select_device_type(platforms, &platformCount, &device_type);
	validate_selection(platforms, &platformCount, ctxprop, &device_type);
	
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, NULL );
	if( !context ) { printf("ERROR: clCreateContextFromType() failed\n"); return -1; }

	// get the list of GPUs
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
//	printf("num_devices = %d\n", num_devices);

	if( result != CL_SUCCESS || num_devices < 1 ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }
	device_list = new cl_device_id[num_devices];
	//device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS ) { printf("ERROR: clGetContextInfo() failed\n"); return -1; }

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context, device_list[0], 0, NULL );
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete[] device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	int seed = 7;

	init_fpga(&argc, &argv, &version);

	if ( setup(argc, argv) != 0)
	{
		return -1;
	}

	bpnn_initialize(seed);
	backprop_face();

	shutdown();

	return 0;
}

int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
	int in, hid, out;
	float out_err, hid_err;

	TimeStamp start, end;
#ifdef PROFILE
	TimeStamp start1, end1;
#endif
  
	in = net->input_n;
	hid = net->hidden_n;
	out = net->output_n;

	// read the kernel core source
	const char * kernel_bp1 = (is_ndrange_kernel(version)) ? "bpnn_layerforward_ocl" : "bpnn_layerforward";
	const char * kernel_bp2 = (is_ndrange_kernel(version)) ? "bpnn_adjust_weights_ocl" : "bpnn_output_error";
	const char * kernel_bp3 = "bpnn_hidden_error";
	const char * kernel_bp4 = "bpnn_adjust_weights";

	size_t sourcesize;
	char *kernel_file_path = getVersionedKernelName("./backprop_kernel", version);
	char *source = read_kernel(kernel_file_path, &sourcesize);
	free(kernel_file_path);

	if(initialize()) return -1;

	// compile kernel
	cl_int err = 0;
#if defined(USE_JIT)
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); return -1; }
#else
	cl_program prog = clCreateProgramWithBinary(context, 1, device_list, &sourcesize, (const unsigned char**)&source, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithBinary() => %d\n", err); return -1; }
#endif

	clBuildProgram_SAFE(prog, num_devices, device_list, NULL, NULL, NULL);

	cl_kernel kernel1 = NULL;
	cl_kernel kernel2 = NULL;
	cl_kernel kernel3 = NULL;
	cl_kernel kernel4 = NULL;
	kernel1 = clCreateKernel(prog, kernel_bp1, &err);
	kernel2 = clCreateKernel(prog, kernel_bp2, &err);
	if (!is_ndrange_kernel(version))
	{
		kernel3 = clCreateKernel(prog, kernel_bp3, &err);
		kernel4 = clCreateKernel(prog, kernel_bp4, &err);
	}
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	clReleaseProgram(prog);

	float *input_weights_one_dim;
	float *input_prev_weights_one_dim;
	float *partial_sum = NULL;
	float *hidden_weights_one_dim = NULL;
	float *hidden_prev_weights_one_dim = NULL;
	float sum;
	int num_blocks = in / BLOCK_SIZE;

	input_weights_one_dim = (float *) alignedMalloc((in + 1) * (hid + 1) * sizeof(float));
	input_prev_weights_one_dim = (float *) alignedMalloc((in + 1) * (hid + 1) * sizeof(float));
	if(is_ndrange_kernel(version))
	{
		partial_sum = (float *) alignedMalloc(num_blocks * WIDTH * sizeof(float));
	}
	else
	{
		hidden_weights_one_dim = (float *) alignedMalloc((hid + 1) * (out + 1) * sizeof(float));
		hidden_prev_weights_one_dim = (float *) alignedMalloc((hid + 1) * (out + 1) * sizeof(float));
	}

	// set global and local workitems
	size_t global_work[3] = { BLOCK_SIZE, BLOCK_SIZE * (size_t)num_blocks, 1 }; 
	size_t local_work[3] = { BLOCK_SIZE, BLOCK_SIZE, 1 };

	// convert 2D buffers to 1D since OpenCl doesn't support 2D buffers
	int m = 0;
	for (int k = 0; k <= in; k++)
	{
		for (int j = 0; j <= hid; j++)
		{
			input_weights_one_dim[m]      = net->input_weights[k][j];
			input_prev_weights_one_dim[m] = net->input_prev_weights[k][j];
			m++;
		}
	}

	if (!is_ndrange_kernel(version))
	{
		m = 0;
		for (int k = 0; k <= hid; k++)
		{
			for (int j = 0; j <= out; j++)
			{
				hidden_weights_one_dim[m]      = net->hidden_weights[k][j];
				hidden_prev_weights_one_dim[m] = net->hidden_prev_weights[k][j];
				m++;
			}
		}
	}

	cl_mem input_hidden_ocl;
	cl_mem input_hidden_ocl2;
	cl_mem input_ocl;
	cl_mem output_hidden_ocl;
	cl_mem output_hidden_ocl2;
	cl_mem hidden_partial_sum;
	cl_mem target_ocl;
	cl_mem hidden_delta_ocl;
	cl_mem output_delta_ocl;
	cl_mem input_prev_weights_ocl;
	cl_mem hidden_prev_weights_ocl;
	cl_mem out_err_ocl;
	cl_mem hid_err_ocl;

	input_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_ocl\n"); return -1;}
	input_hidden_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_hidden_ocl\n"); return -1;}
	output_hidden_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_hidden_ocl\n"); return -1;}
	hidden_delta_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hidden_delta_ocl\n"); return -1;}
	input_prev_weights_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_prev_weights_ocl\n"); return -1;}
	if (is_ndrange_kernel(version))
	{
		hidden_partial_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, num_blocks * WIDTH * sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hidden_partial_sum\n"); return -1;}
	}
	else
	{
		input_hidden_ocl2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * (out + 1) * sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_hidden_ocl2\n"); return -1;}
		output_hidden_ocl2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_hidden_ocl2\n"); return -1;}
		target_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (out + 1) * sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer target_ocl\n"); return -1;}
		output_delta_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (out + 1) * sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_delta_ocl\n"); return -1;}
		hidden_prev_weights_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * (out + 1) * sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hidden_prev_weights_ocl\n"); return -1;}
		out_err_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer out_err_ocl\n"); return -1;}
		hid_err_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float), NULL, &err );
		if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hid_err_ocl\n"); return -1;}
	}

	printf("Computing...\n");

	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, input_ocl, 1, 0, (in + 1) * sizeof(float), net->input_units, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_ocl\n"); return -1; }
	err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, 0);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_hidden_ocl\n"); return -1; }
	if (!is_ndrange_kernel(version))
	{
		err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl2, 1, 0, (hid + 1) * (out + 1) * sizeof(float), hidden_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_hidden_ocl2\n"); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, target_ocl, 1, 0, (out + 1) * sizeof(float), net->target, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer target_ocl\n"); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, hidden_prev_weights_ocl, 1, 0, (hid + 1) * (out + 1) * sizeof(float), hidden_prev_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer hidden_prev_weights_ocl\n"); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, input_prev_weights_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_prev_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_prev_weights_ocl\n"); return -1; }
	}

	if (is_ndrange_kernel(version))
	{
		GetTime(start);
		CL_SAFE_CALL( clSetKernelArg(kernel1, 0, sizeof(void *)                , (void*) &input_ocl          ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 1, sizeof(void *)                , (void*) &output_hidden_ocl  ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 2, sizeof(void *)                , (void*) &input_hidden_ocl   ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 3, sizeof(void *)                , (void*) &hidden_partial_sum ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 4, sizeof(float) * HEIGHT        , (void*) NULL                ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 5, sizeof(float) * HEIGHT * WIDTH, (void*) NULL                ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 6, sizeof(cl_int)                , (void*) &in                 ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 7, sizeof(cl_int)                , (void*) &hid                ) );

		err = clEnqueueNDRangeKernel(cmd_queue, kernel1, 2, NULL, global_work, local_work, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }

		err = clFinish(cmd_queue);
		if(err != CL_SUCCESS) { printf("ERROR: 1  clFinish()=>%d failed\n", err); return -1; }

		err = clEnqueueReadBuffer(cmd_queue, hidden_partial_sum, 1, 0, num_blocks * WIDTH * sizeof(float), partial_sum, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueReadBuffer: partial sum\n"); return -1; }

		for (int j = 1; j <= hid; j++) {
			sum = 0.0;
			for (int k = 0; k < num_blocks; k++) {
				sum += partial_sum[k * hid + j-1] ;
			}
			sum += net->input_weights[0][j];
			net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
		}

		bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
		bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
		bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);  
		bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

		err = clEnqueueWriteBuffer(cmd_queue, hidden_delta_ocl,       1, 0, (hid + 1) * sizeof(float), net->hidden_delta, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer hidden_delta_ocl\n"); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, input_prev_weights_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_prev_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_prev_weights_ocl\n"); return -1; }
		err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl,       1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_hidden_ocl\n"); return -1; }

		CL_SAFE_CALL( clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &hidden_delta_ocl       ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*) &hid                    ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &input_ocl              ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &in                     ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 4, sizeof(void *), (void*) &input_hidden_ocl       ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 5, sizeof(void *), (void*) &input_prev_weights_ocl ) );

		err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 2, NULL, global_work, local_work, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 2  clEnqueueNDRangeKernel()=>%d failed\n", err); return -1; }
	}
	else
	{
		GetTime(start);
		CL_SAFE_CALL( clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &input_ocl         ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &output_hidden_ocl ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &input_hidden_ocl  ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &in                ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 4, sizeof(cl_int), (void*) &hid               ) );

		#ifdef PROFILE
		GetTime(start1);
		#endif

		CL_SAFE_CALL( clEnqueueTask(cmd_queue, kernel1, 0, NULL, NULL) );

		#ifdef PROFILE
		clFinish(cmd_queue);
		GetTime(end1);
		printf("Kernel 1: %f ms\n", TimeDiff(start1, end1));
		#endif

		CL_SAFE_CALL( clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &output_hidden_ocl ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &output_hidden_ocl2) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &input_hidden_ocl2 ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 3, sizeof(cl_int), (void*) &hid               ) );
		CL_SAFE_CALL( clSetKernelArg(kernel1, 4, sizeof(cl_int), (void*) &out               ) );

		#ifdef PROFILE
		GetTime(start1);
		#endif

		CL_SAFE_CALL( clEnqueueTask(cmd_queue, kernel1, 0, NULL, NULL) );

		#ifdef PROFILE
		clFinish(cmd_queue);
		GetTime(end1);
		printf("Kernel 2: %f ms\n", TimeDiff(start1, end1));
		#endif

		CL_SAFE_CALL( clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &output_delta_ocl  ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &target_ocl        ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &output_hidden_ocl2) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &out               ) );
		CL_SAFE_CALL( clSetKernelArg(kernel2, 4, sizeof(void *), (void*) &out_err_ocl       ) );

		#ifdef PROFILE
		GetTime(start1);
		#endif

		CL_SAFE_CALL( clEnqueueTask(cmd_queue, kernel2, 0, NULL, NULL) );

		#ifdef PROFILE
		clFinish(cmd_queue);
		GetTime(end1);
		printf("Kernel 3: %f ms\n", TimeDiff(start1, end1));
		#endif

		CL_SAFE_CALL( clSetKernelArg(kernel3, 0, sizeof(void *), (void*) &hidden_delta_ocl ) );
		CL_SAFE_CALL( clSetKernelArg(kernel3, 1, sizeof(cl_int), (void*) &hid              ) );
		CL_SAFE_CALL( clSetKernelArg(kernel3, 2, sizeof(void *), (void*) &output_delta_ocl ) );
		CL_SAFE_CALL( clSetKernelArg(kernel3, 3, sizeof(cl_int), (void*) &out              ) );
		CL_SAFE_CALL( clSetKernelArg(kernel3, 4, sizeof(void *), (void*) &input_hidden_ocl2) );
		CL_SAFE_CALL( clSetKernelArg(kernel3, 5, sizeof(void *), (void*) &output_hidden_ocl) );
		CL_SAFE_CALL( clSetKernelArg(kernel3, 6, sizeof(void *), (void*) &hid_err_ocl      ) );

		#ifdef PROFILE
		GetTime(start1);
		#endif

		CL_SAFE_CALL( clEnqueueTask(cmd_queue, kernel3, 0, NULL, NULL) );

		#ifdef PROFILE
		clFinish(cmd_queue);
		GetTime(end1);
		printf("Kernel 4: %f ms\n", TimeDiff(start1, end1));
		#endif

		CL_SAFE_CALL( clSetKernelArg(kernel4, 0, sizeof(void *), (void*) &output_delta_ocl       ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 1, sizeof(cl_int), (void*) &out                    ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 2, sizeof(void *), (void*) &output_hidden_ocl      ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 3, sizeof(cl_int), (void*) &hid                    ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 4, sizeof(void *), (void*) &input_hidden_ocl2      ) ); //read back
		CL_SAFE_CALL( clSetKernelArg(kernel4, 5, sizeof(void *), (void*) &hidden_prev_weights_ocl) );

		#ifdef PROFILE
		GetTime(start1);
		#endif

		CL_SAFE_CALL( clEnqueueTask(cmd_queue, kernel4, 0, NULL, NULL) );

		#ifdef PROFILE
		clFinish(cmd_queue);
		GetTime(end1);
		printf("Kernel 5: %f ms\n", TimeDiff(start1, end1));
		#endif

		CL_SAFE_CALL( clSetKernelArg(kernel4, 0, sizeof(void *), (void*) &hidden_delta_ocl       ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 1, sizeof(cl_int), (void*) &hid                    ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 2, sizeof(void *), (void*) &input_ocl              ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 3, sizeof(cl_int), (void*) &in                     ) );
		CL_SAFE_CALL( clSetKernelArg(kernel4, 4, sizeof(void *), (void*) &input_hidden_ocl       ) ); // read back
		CL_SAFE_CALL( clSetKernelArg(kernel4, 5, sizeof(void *), (void*) &input_prev_weights_ocl ) );

		#ifdef PROFILE
		GetTime(start1);
		#endif

		CL_SAFE_CALL( clEnqueueTask(cmd_queue, kernel4, 0, NULL, NULL) );

		#ifdef PROFILE
		clFinish(cmd_queue);
		GetTime(end1);
		printf("Kernel 6: %f ms\n", TimeDiff(start1, end1));
		#endif
	}

	err = clFinish(cmd_queue);
	if(err != CL_SUCCESS) { printf("ERROR: 2  clFinish()=>%d failed\n", err); return -1; }
	GetTime(end);

	if (is_ndrange_kernel(version))
	{
		err = clEnqueueReadBuffer(cmd_queue, input_ocl, 1, 0, (in + 1) * sizeof(float), net->input_units, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 2  clEnqueueReadBuffer: input_ocl\n"); return -1; }
		err = clEnqueueReadBuffer(cmd_queue, input_hidden_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 2  clEnqueueReadBuffer: input_hidden_ocl\n"); return -1; }
	}
	else
	{
		err = clEnqueueReadBuffer(cmd_queue, input_hidden_ocl2, 1, 0, (hid + 1) * (out + 1) * sizeof(float), hidden_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 2  clEnqueueReadBuffer: input_hidden_ocl2\n"); return -1; }
		err = clEnqueueReadBuffer(cmd_queue, input_hidden_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, 0);
		if(err != CL_SUCCESS) { printf("ERROR: 2  clEnqueueReadBuffer: input_hidden_ocl\n"); return -1; }
	}

	// convert 1D buffers to 2D
	if (!is_ndrange_kernel(version))
	{
		m = 0;
		for (int k = 0; k <= in; k++)
		{
			for (int j = 0; j <= hid; j++)
			{
				net->input_weights[k][j] = input_weights_one_dim[m];
				m++;
			}
		}

		m = 0;
		for (int k = 0; k <= hid; k++)
		{
			for (int j = 0; j <= out; j++)
			{
				net->hidden_weights[k][j] = hidden_weights_one_dim[m];
				m++;
			}
		}
	}
  
	clReleaseMemObject(input_ocl);
	clReleaseMemObject(output_hidden_ocl);
	clReleaseMemObject(input_hidden_ocl);
	clReleaseMemObject(input_prev_weights_ocl);
	if(is_ndrange_kernel(version))
	{
		clReleaseMemObject(hidden_partial_sum);
	}
	else
	{
		clReleaseMemObject(input_hidden_ocl2);
		clReleaseMemObject(output_hidden_ocl2);
		clReleaseMemObject(target_ocl);
		clReleaseMemObject(output_delta_ocl);
		clReleaseMemObject(hidden_prev_weights_ocl);
	}
  
	free(input_prev_weights_one_dim);
	free(input_weights_one_dim);
	if(is_ndrange_kernel(version))
	{
		free(partial_sum);
	}
	else
	{
		free(hidden_weights_one_dim);
		free(hidden_prev_weights_one_dim);
	}

#ifdef OUTPUT
	bpnn_save(net, "output.txt");
#endif

	printf("Computation done in %0.3lf ms.\n", TimeDiff(start, end));

	return 0;
}
