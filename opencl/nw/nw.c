#include "work_group_size.h"

#define LIMIT -999

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <assert.h>

#ifdef NV //NVIDIA
	#include <oclUtils.h>
#elif __APPLE__
	#include <OpenCL/cl.h>
#else 
	#include <CL/cl.h>
#endif

#include "../common/opencl_util.h"
#include "../../common/timer.h"
#include "../common/opencl_timer.h"

//global variables

int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

// local variables
static cl_context       context;
static cl_command_queue cmd_queue;
static cl_command_queue cmd_queue2;
static cl_device_type   device_type;
static cl_device_id   * device_list;
static cl_int           num_devices;

static int initialize()
{
	size_t size;
	cl_int result, error;
	cl_uint platformCount;
	cl_platform_id* platforms = NULL;
	cl_context_properties ctxprop[3];

	display_device_info(&platforms, &platformCount);
	select_device_type(platforms, &platformCount, &device_type);
	validate_selection(platforms, &platformCount, ctxprop, &device_type);
	
	// create OpenCL context
	context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, &error );
	if( !context )
	{
		printf("ERROR: clCreateContextFromType(%s) failed with error code %d.\n", (device_type == CL_DEVICE_TYPE_ACCELERATOR) ? "FPGA" : (device_type == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", error);
                display_error_message(error, stdout);
		return -1;
	}

	// get the list of GPUs
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &size );
	num_devices = (int) (size / sizeof(cl_device_id));
	
	if( result != CL_SUCCESS || num_devices < 1 )
	{
		printf("ERROR: clGetContextInfo() failed\n");
		return -1;
	}
	device_list = new cl_device_id[num_devices];
	if( !device_list )
	{
		printf("ERROR: new cl_device_id[] failed\n");
		return -1;
	}
	result = clGetContextInfo( context, CL_CONTEXT_DEVICES, size, device_list, NULL );
	if( result != CL_SUCCESS )
	{
		printf("ERROR: clGetContextInfo() failed\n");
		return -1;
	}

	// create command queue for the first device
	cmd_queue = clCreateCommandQueue( context, device_list[0], CL_QUEUE_PROFILING_ENABLE, NULL );
	if( !cmd_queue )
	{
		printf("ERROR: clCreateCommandQueue() failed\n");
		return -1;
	}

	// create command queue for the first device
	cmd_queue2 = clCreateCommandQueue( context, device_list[0], CL_QUEUE_PROFILING_ENABLE, NULL );
	if( !cmd_queue )
	{
		printf("ERROR: clCreateCommandQueue() failed\n");
		return -1;
	}
	
	free(platforms); // platforms isn't needed in the main function

	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = 0;

	return 0;
}

int maximum( int a,
		 int b,
		 int c){

	int k;
	if( a <= b )
	  k = b;
	else 
	  k = a;
	if( k <=c )
	  return(c);
	else
	  return(k);
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> <kernel_version> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	fprintf(stderr, "\t<kernel_version> - version of kernel or bianry file to load\n");
	exit(1);
}

int main(int argc, char **argv)
{
	int max_rows, max_cols, penalty;
        char *version_string;
        int version_number;
	size_t sourcesize;
        int block_size = BSIZE;

        init_fpga2(&argc, &argv, &version_string, &version_number);

	// for calculating execution time
	double time = 0;
        int enable_traceback = 0;

	if (argc >= 3) {
          max_rows = atoi(argv[1]);
          max_cols = atoi(argv[1]);
          penalty = atoi(argv[2]);
        } else {
          usage(argc, argv);
          exit(1);
	}

        // The default block size is BLOCK_SIZE. Can be configured
        // with a command line parameter.
        if (argc >= 4) {
          block_size = atoi(argv[3]);
        }

        if (argc >= 5) {
          enable_traceback = atoi(argv[4]);
        }
	
	// the lengths of the two sequences should be divisible by 16.
	// And at current stage, max_rows needs to equal max_cols
	if(atoi(argv[1])%16!=0){
	fprintf(stderr,"The dimension values must be a multiple of 16\n");
	exit(1);
	}

        max_rows = max_rows + 1;
        max_cols = max_cols + 1;
        
        int off1 = (version_number != 9) ? 1 : 0;
        int offm1 = (version_number != 9) ? 0 : 1;        
        int mc1 = (version_number != 9) ? max_cols : max_cols - 1;
        int mr1 = (version_number != 9) ? max_rows : max_rows - 1;
        size_t ref_size = mc1 * mr1 * sizeof(int);
        int *reference = (int *)alignedMalloc(ref_size);
	int *input_itemsets = (int *)alignedMalloc( max_rows * max_cols * sizeof(int) );
        int *input_itemsets_h = (int *)alignedMalloc( mc1 * sizeof(int) );
        int *input_itemsets_v = (int *)alignedMalloc( mr1 * sizeof(int) );

        size_t output_itemsets_size =  mc1 * mr1 * sizeof(int);
	int *output_itemsets = (int *)alignedMalloc(output_itemsets_size);
	
	srand(7);
	
	//initialization 
	for (int i = 0 ; i < max_cols; i++){
		for (int j = 0 ; j < max_rows; j++){
			input_itemsets[i*max_cols+j] = 0;
		}
	}

	for( int i=1; i< max_rows ; i++){    //initialize the cols
			input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
	
	for( int j=1; j< max_cols ; j++){    //initialize the rows
			input_itemsets[j] = rand() % 10 + 1;
	}
	
	for (int i = 1 ; i < max_cols; i++){
		for (int j = 1 ; j < max_rows; j++){
                  int ref_offset = (version_number == 9) ?
                      (i-1) * mc1 + (j-1) : i * mc1 + j;
                  reference[ref_offset] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

	for( int i = 1; i< max_rows ; i++) {
          input_itemsets_v[i-offm1] = input_itemsets[i*max_cols] = -i * penalty;
        }
	for( int j = 1; j< max_cols ; j++) {
          input_itemsets_h[j-offm1] = input_itemsets[j] = -j * penalty;
        }

	// get name of kernel file based on version
	char *kernel_file_path = getVersionedKernelName2("./nw_kernel",
                                                         version_string);

	char *source = read_kernel(kernel_file_path, &sourcesize);

	// read the kernel core source
	char const * kernel_nw1  = "nw_kernel1";
	char const * kernel_nw2  = "nw_kernel2";

	int nworkitems, workgroupsize = 0;
	nworkitems = block_size;

	if(nworkitems < 1 || workgroupsize < 0){
		printf("ERROR: invalid or missing <num_work_items>[/<work_group_size>]\n"); 
		return -1;
	}
	// set global and local workitems
	size_t local_work[3] = { (size_t)((workgroupsize>0)?workgroupsize:1), 1, 1 };
	size_t global_work[3] = { (size_t)nworkitems, 1, 1 }; //nworkitems = no. of GPU threads
	
	// OpenCL initialization
	if(initialize())
	{
		return -1;
	}

        printf("block_size: %d\n", block_size);

	// compile kernel
	cl_int err = 0;
#ifdef USE_JIT
	const char * slist[2] = { source, 0 };
	cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
#else
	cl_program prog = clCreateProgramWithBinary(context, 1, device_list,
                                                    &sourcesize, (const unsigned char**)&source, NULL, &err);
#endif
	if(err != CL_SUCCESS) {
          printf("ERROR: clCreateProgramWithSource/Binary() => %d\n", err);
          display_error_message(err, stderr);
          return -1;
        }

	char clOptions[110];
        sprintf(clOptions, "-I .");

#ifdef USE_JIT
#ifdef USE_RESTRICT
        sprintf(clOptions + strlen(clOptions), " -DUSE_RESTRICT");
#endif  
	sprintf(clOptions + strlen(clOptions), " -DBSIZE=%d", block_size);
#endif
        
	clBuildProgram_SAFE(prog, num_devices, device_list, clOptions, NULL, NULL);

	cl_kernel kernel1;
	cl_kernel kernel2;
	kernel1 = clCreateKernel(prog, kernel_nw1, &err);
        if (version_number == 11 || version_number == 15 ||
            (is_ndrange_kernel(version_number) && version_number != 4)) {
          kernel2 = clCreateKernel(prog, kernel_nw2, &err);
        } else {
          // use the same kernel in single work-item versions
          kernel2 = kernel1;
        }
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); return -1; }
	clReleaseProgram(prog);
	
	// creat buffers
	cl_mem input_itemsets_d;
	cl_mem output_itemsets_d;
	cl_mem reference_d;
	
	input_itemsets_d = clCreateBuffer(context, CL_MEM_READ_WRITE, max_cols * max_rows * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_item_set (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	reference_d		 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                  ref_size, NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer reference (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
	output_itemsets_d = clCreateBuffer(
            context, CL_MEM_READ_WRITE, mc1 * mr1 * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_item_set (size:%d) => %d\n", max_cols * max_rows, err); return -1;}
        cl_mem input_itemsets_h_d = clCreateBuffer(
            context, CL_MEM_READ_WRITE, mc1 * sizeof(int), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_item_set_h (size:%d) => %d\n", max_cols, err); return -1;}

        cl_mem output_itemsets_h_d = clCreateBuffer(
            context, CL_MEM_READ_WRITE, max_cols * sizeof(int), NULL, &err );

        // Create enough vertical vectors
        int num_bridge_columns = (max_cols / block_size)  + 1;
        cl_mem *input_itemsets_v_d = (cl_mem*)malloc(
            sizeof(cl_mem) * num_bridge_columns);
        for (int i = 0; i < num_bridge_columns; ++i) {
          input_itemsets_v_d[i] = clCreateBuffer(
              context, CL_MEM_READ_WRITE, mr1 * sizeof(int), NULL, &err );
          if(err != CL_SUCCESS) {
            printf("ERROR: clCreateBuffer input_item_set_v[0] (size:%d) => %d\n", max_rows, err);
            return -1;
          }
        }
        
	//write buffers
	CL_SAFE_CALL(clEnqueueWriteBuffer(
            cmd_queue, input_itemsets_d, 1,
            0, max_cols * max_rows * sizeof(int),
            input_itemsets, 0, 0, 0));
        // copy input_itemsets to to output_itemsets to initialize the
        // border areas
#if 0        
	CL_SAFE_CALL(clEnqueueWriteBuffer(
            cmd_queue, output_itemsets_d, 1,
            0, max_cols * max_rows * sizeof(int),
            input_itemsets, 0, 0, 0));
#endif        
	CL_SAFE_CALL(clEnqueueWriteBuffer(
            cmd_queue, reference_d, 1, 0,
            ref_size,
            reference, 0, 0, 0));
	CL_SAFE_CALL(clEnqueueWriteBuffer(
            cmd_queue, input_itemsets_h_d, 1, 0,
            mc1 * sizeof(int),
            input_itemsets_h, 0, 0, 0));
        CL_SAFE_CALL(clEnqueueWriteBuffer(
            cmd_queue, input_itemsets_v_d[0], 1, 0,
            mr1 * sizeof(int),
            input_itemsets_v, 0, 0, 0));

		
	int worksize = max_cols - 1;
	printf("WG size of kernel = %d \n", block_size);
	printf("worksize = %d\n", worksize);
	//these two parameters are for extension use, don't worry about it.
	int offset_r = 0, offset_c = 0;
	int block_width = worksize/block_size ;
        
        int kernel1_arg_idx = 0;
        
        // the first argument is updated at every call for the ndrange kernels
        if (is_ndrange_kernel(version_number)) {
          ++kernel1_arg_idx;
        }
        
        clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(void *), (void*) &reference_d);
        if (is_ndrange_kernel(version_number) || version_number < 7) {
          clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(void *), (void*) &input_itemsets_d);
        }

#if 0
        // We don't maintain version 5 anymore. The kernel is
        // available in nw_kernel_single_work_item_2d_blocking.cl
        if (version_number == 5) {
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void *), (void*) &input_itemsets_d);
        }
#endif        
        if (version_number == 7 || version_number == 9) {
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void *), (void*) &input_itemsets_h_d);
          kernel1_arg_idx++;
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void *), (void*) &output_itemsets_d);
          kernel1_arg_idx++;
        }

        if (version_number == 11) {
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void *), (void*) &input_itemsets_h_d);
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void*), (void*) &(input_itemsets_v_d[0]));
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void *), (void*) &output_itemsets_d);
        }

        if (version_number == 15) {
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void *), (void*) &input_itemsets_h_d);
          kernel1_arg_idx++;
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(void *), (void*) &output_itemsets_d);
          kernel1_arg_idx++;
          clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(void *),
                         (void*) &output_itemsets_h_d);
        }
        
        if (is_ndrange_kernel(version_number)) {
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(cl_int) * (block_size + 1) *(block_size+1),
                         (void*)NULL );
          clSetKernelArg(kernel1, kernel1_arg_idx++,
                         sizeof(cl_int) *  block_size * block_size, (void*)NULL );

        }
        
        clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(cl_int), (void*) &mr1);
        clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(cl_int), (void*) &penalty);

        if (version_number == 4) {
          clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(cl_int), (void*) &block_width);
        }
        
        if (is_ndrange_kernel(version_number)) {
          clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(cl_int), (void*) &offset_r);
          clSetKernelArg(kernel1, kernel1_arg_idx++, sizeof(cl_int), (void*) &offset_c);
        }

        if (is_ndrange_kernel(version_number) && version_number <= 2) {
          clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &reference_d);
          clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &input_itemsets_d);
          clSetKernelArg(kernel2, 2, sizeof(cl_int) * (block_size + 1) *(block_size+1), (void*)NULL );
          clSetKernelArg(kernel2, 3, sizeof(cl_int) * block_size *block_size, (void*)NULL );
          clSetKernelArg(kernel2, 4, sizeof(cl_int), (void*) &max_cols);
          clSetKernelArg(kernel2, 5, sizeof(cl_int), (void*) &penalty);
          clSetKernelArg(kernel2, 7, sizeof(cl_int), (void*) &block_width);
          clSetKernelArg(kernel2, 8, sizeof(cl_int), (void*) &offset_r);
          clSetKernelArg(kernel2, 9, sizeof(cl_int), (void*) &offset_c);
        }

        if (version_number == 11) {
          clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &reference_d);
          clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &input_itemsets_h_d);
          clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &output_itemsets_d);
          clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &max_cols);
          clSetKernelArg(kernel2, 4, sizeof(cl_int), (void*) &penalty);
        }

        if (version_number == 15) {
          clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &reference_d);
          clSetKernelArg(kernel2, 1, sizeof(void *), (void*) &output_itemsets_h_d);
          clSetKernelArg(kernel2, 3, sizeof(void *), (void*) &output_itemsets_d);
          clSetKernelArg(kernel2, 5, sizeof(cl_int), (void*) &max_cols);
          clSetKernelArg(kernel2, 6, sizeof(cl_int), (void*) &penalty);
        }

	// for calculating runtime
        cl_event * kernel_event1 = NULL; 
	cl_event * kernel_event2 = (cl_event *)malloc(sizeof(cl_event) * worksize);

        if (!is_ndrange_kernel(version_number) && version_number <= 3) {
          kernel_event1 = (cl_event *)malloc(sizeof(cl_event));
          CL_SAFE_CALL(clEnqueueTask(
              cmd_queue, kernel1, 0, NULL, &kernel_event1[0]));
          clFinish(cmd_queue);          
        } else if (version_number == 5) {
          int nb = (max_cols - 1)  / block_size;
          kernel_event1 = (cl_event *)malloc(sizeof(cl_event) * nb);
          for (int bx = 0; bx < nb; ++bx) {
            clSetKernelArg(kernel1, kernel1_arg_idx, sizeof(cl_int), (void*) &bx);
            CL_SAFE_CALL(clEnqueueTask(
                cmd_queue, kernel1, 0, NULL, &kernel_event1[bx]));
          }
          clFinish(cmd_queue);
        } else if (version_number == 7 || version_number == 9) {
          int nb = (max_cols - 1)  / block_size;
          printf("nb: %d\n", nb);
          kernel_event1 = (cl_event *)malloc(sizeof(cl_event) * nb);
          int vertical_input_idx = 0;
          for (int bx = 0; bx < nb; ++bx) {
            clSetKernelArg(kernel1, 2, sizeof(void*), (void*) &(input_itemsets_v_d[vertical_input_idx]));
            clSetKernelArg(kernel1, 4, sizeof(void*), (void*) &(input_itemsets_v_d[vertical_input_idx ^ 1]));
            clSetKernelArg(kernel1, kernel1_arg_idx, sizeof(cl_int), (void*) &bx);
            CL_SAFE_CALL(clEnqueueTask(
                cmd_queue, kernel1, 0, NULL, &kernel_event1[bx]));
            vertical_input_idx ^= 1;
          }
          clFinish(cmd_queue);
        } else if (version_number == 11) {
          int nb = (max_cols - 1)  / block_size;
          assert(nb % 2 == 0);
          kernel_event1 = (cl_event *)malloc(sizeof(cl_event) * nb);
          TimeStamp start, end;
          GetTime(start);
          for (int bx = 0; bx < nb;) {
            clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &bx);
            CL_SAFE_CALL(clEnqueueTask(
                cmd_queue, kernel1, 0, NULL, &kernel_event1[bx]));
            ++bx;
            clSetKernelArg(kernel2, 5, sizeof(cl_int), (void*) &bx);
            CL_SAFE_CALL(clEnqueueTask(
                cmd_queue2, kernel2, 0, NULL, &kernel_event1[bx]));
            ++bx;
          }
          clFinish(cmd_queue);
          clFinish(cmd_queue2);
          GetTime(end);
          printf("Computation done in %0.3lf ms.\n", TimeDiff(start, end));
        } else if (version_number == 15) {
          int nb = (max_cols - 1)  / block_size;
          assert(nb % 2 == 0);
          kernel_event1 = (cl_event *)malloc(sizeof(cl_event) * nb * 2);
          TimeStamp start, end;
          GetTime(start);
          for (int bx = 0; bx < nb; ++bx) {
            clSetKernelArg(kernel1, 2, sizeof(void*),
                           (void*) &(input_itemsets_v_d[bx]));
            clSetKernelArg(kernel1, 4, sizeof(void*),
                           (void*) &(input_itemsets_v_d[bx+1]));
            clSetKernelArg(kernel1, 8, sizeof(cl_int), (void*) &bx);
            CL_SAFE_CALL(clEnqueueTask(
                cmd_queue, kernel1, 0, NULL, &kernel_event1[bx*2]));
            clSetKernelArg(kernel2, 2, sizeof(void*),
                           (void*) &(input_itemsets_v_d[bx]));
            clSetKernelArg(kernel2, 4, sizeof(void*),
                           (void*) &(input_itemsets_v_d[bx+1]));
            clSetKernelArg(kernel2, 7, sizeof(cl_int), (void*) &bx);
            CL_SAFE_CALL(clEnqueueTask(
                cmd_queue2, kernel2, 1, &kernel_event1[bx*2],
                &kernel_event1[bx*2+1]));
          }
          clFinish(cmd_queue2);
          GetTime(end);
          printf("Computation done in %0.3lf ms.\n", TimeDiff(start, end));
        } else {
          // NDRange versions
          printf("Processing upper-left matrix\n");          
          kernel_event1 = (cl_event *)malloc(sizeof(cl_event) * worksize);          
          if (is_ndrange_kernel(version_number)) {
            if (version_number == 4) {
              int is_upper_left = 1;
              clSetKernelArg(kernel1, kernel1_arg_idx, sizeof(cl_int), (void*) &is_upper_left);
            }
            for( int blk = 1 ; blk <= worksize/block_size ; blk++){
              global_work[0] = block_size * blk;
              local_work[0]  = block_size;
              clSetKernelArg(kernel1, 0, sizeof(cl_int), (void*) &blk);
              CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel1, 2, NULL,
                                                   global_work, local_work, 0, 0, &kernel_event1[blk]) );
            }
          } else {
            assert(0 && "Unused?");
            int top_left = 1;
            clSetKernelArg(kernel1, kernel1_arg_idx, sizeof(cl_int), (void*) &top_left);
            for( int i = 0 ; i < max_cols-2 ; i++){
              clSetKernelArg(kernel1, 0, sizeof(cl_int), (void*) &i);
              CL_SAFE_CALL(clEnqueueTask(
                  cmd_queue, kernel1, 0, NULL, &kernel_event1[i]));
            }
          }
          clFinish(cmd_queue);
	
          printf("Processing lower-right matrix\n");
          if (is_ndrange_kernel(version_number)) {
            if (version_number == 4) {
              int is_upper_left = 0;
              clSetKernelArg(kernel1, kernel1_arg_idx, sizeof(cl_int), (void*) &is_upper_left);
            }
            for( int blk =  worksize/block_size - 1  ; blk >= 1 ; blk--){
              global_work[0] = block_size * blk;
              local_work[0] =  block_size;
              clSetKernelArg(kernel2, version_number == 4 ? 0 : 6, sizeof(cl_int), (void*) &blk);
              CL_SAFE_CALL( clEnqueueNDRangeKernel(cmd_queue, kernel2, 2, NULL,
                                                   global_work, local_work, 0, 0, &kernel_event2[blk]) );
            }
          } else {
            assert(0 && "Unused?");            
            int top_left = 0;
            clSetKernelArg(kernel1, kernel1_arg_idx, sizeof(cl_int), (void*) &top_left);
            for( int i = max_cols - 4 ; i >= 0 ; i--){          
              clSetKernelArg(kernel1, 0, sizeof(cl_int), (void*) &i);
              CL_SAFE_CALL(clEnqueueTask(
                  cmd_queue, kernel1, 0, NULL, &kernel_event2[i]));
            }
          }
          clFinish(cmd_queue);
        }
	
	// time calculation
        if (!is_ndrange_kernel(version_number)) {
          int num_launches = 0;
          int nb = (max_cols-1) / block_size;          
          switch (version_number) {
            case 1:
            case 3:
              num_launches = 1;
              break;
            case 5:
            case 7:
            case 9:              
            case 11:
              num_launches = nb;
              break;
            case 15:
              num_launches = nb*2;
              break;
            default:
              abort();
          }
          time = 0;
          printf("First kernel: %f ms\n",
                 CLTimeDiff( CLGetTime(kernel_event1[0], START),
                             CLGetTime(kernel_event1[0], END) ));
          if (version_number == 9 || version_number == 13) {
            time += CLTimeDiff( CLGetTime(kernel_event1[0], START),
                                CLGetTime(kernel_event1[num_launches-1], END) );
#if 0            
            printf("%lu\n", CLGetTime(kernel_event1[0], START));
            printf("%lu\n", CLGetTime(kernel_event1[0], END));
            printf("%lu\n", CLGetTime(kernel_event1[1], START));
            printf("%lu\n", CLGetTime(kernel_event1[1], END));
            printf("%lu\n", CLGetTime(kernel_event1[2], START));
            printf("%lu\n", CLGetTime(kernel_event1[2], END));
            printf("%lu\n", CLGetTime(kernel_event1[3], START));
            printf("%lu\n", CLGetTime(kernel_event1[3], END));
#endif            
          } else {
            printf("Wall clock: %f\n",
                   CLTimeDiff( CLGetTime(kernel_event1[0], START),
                               CLGetTime(kernel_event1[num_launches-1], END)));
            for (int i = 0; i < num_launches; ++i) {
              time += CLTimeDiff( CLGetTime(kernel_event1[i], START),
                                  CLGetTime(kernel_event1[i], END) );
            }
          }
        } else {
          for( int blk = 1 ; blk <= worksize/block_size ; blk++){
            time += CLTimeDiff( CLGetTime(kernel_event1[blk], START), CLGetTime(kernel_event1[blk], END) );
          }
          for( int blk =  worksize/block_size - 1  ; blk >= 1 ; blk--){
            time += CLTimeDiff( CLGetTime(kernel_event2[blk], START), CLGetTime(kernel_event2[blk], END) );
          }
        }

        if (version_number == 7 || version_number == 9 || version_number == 11 || version_number == 15) {
          err = clEnqueueReadBuffer(cmd_queue, output_itemsets_d, 1, 0,
                                    output_itemsets_size, output_itemsets, 0, 0, 0);
        } else {
          err = clEnqueueReadBuffer(cmd_queue, input_itemsets_d, 1, 0, max_cols * max_rows * sizeof(int), output_itemsets, 0, 0, 0);
        }
	clFinish(cmd_queue);

	printf("Computation done in %0.3lf ms.\n", time);
	printf("==============================================================\n");
        
        FILE *fout = fopen("output_itemsets.txt","w");
        for (int i = off1; i < mr1 - 2; ++i) {
          for (int j = off1; j < mc1 - 2; ++j) {
            fprintf(fout, "[%d, %d] = %d (ref: %d)\n",
                    i-off1, j-off1, output_itemsets[i*mc1 + j],
                    reference[i*mc1+j]);
          }
        }
        fclose(fout);
        printf("Output itemsets saved in output_itemsets.txt\n");
        
        if (enable_traceback) {
          FILE *fpo = fopen("result.txt","w");
          fprintf(fpo, "max_cols: %d, penalty: %d\n", max_cols - 1, penalty);
          for (int i = max_cols - 2,  j = max_rows - 2; i>=0 && j>=0;){
            fprintf(fpo, "[%d, %d] ", i, j);
            int nw = 0, n = 0, w = 0, traceback;
            if ( i == 0 && j == 0 ) {
              fprintf(fpo, "(output: %d)\n", output_itemsets[0]);
              break;
            }
            if ( i > 0 && j > 0 ){
              nw = output_itemsets[(i - 1) * max_cols + j - 1];
              w  = output_itemsets[ i * max_cols + j - 1 ];
              n  = output_itemsets[(i - 1) * max_cols + j];
              fprintf(fpo, "(nw: %d, w: %d, n: %d, ref: %d) ",
                      nw, w, n, reference[i*max_cols+j]);
            }
            else if ( i == 0 ){
              nw = n = LIMIT;
              w  = output_itemsets[ i * max_cols + j - 1 ];
            }
            else if ( j == 0 ){
              nw = w = LIMIT;
              n = output_itemsets[(i - 1) * max_cols + j];
            }
            else{
            }

            //traceback = maximum(nw, w, n);
            int new_nw, new_w, new_n;
            new_nw = nw + reference[i * max_cols + j];
            new_w = w - penalty;
            new_n = n - penalty;
		
            traceback = maximum(new_nw, new_w, new_n);
            if (traceback != output_itemsets[i*max_cols+j]) {
              fprintf(stderr, "Mismatch at (%d, %d). traceback: %d, output_itemsets: %d\n",
                      i, j, traceback, output_itemsets[i*max_cols+j]);
              //exit(1);
            }
            fprintf(fpo, "(output: %d)", traceback);
            
            if(traceback == new_nw) {
              traceback = nw;
              fprintf(fpo, "(->nw) ");
            } else if(traceback == new_w) {
              traceback = w;
              fprintf(fpo, "(->w) ");              
            } else if(traceback == new_n) {
              traceback = n;
              fprintf(fpo, "(->n) ");              
            } else {
              fprintf(stderr, "Error: inconsistent traceback at (%d, %d)\n", i, j);
              abort();
            }
			
            fprintf(fpo, "\n");

            if(traceback == nw )
            {i--; j--; continue;}

            else if(traceback == w )
            {j--; continue;}

            else if(traceback == n )
            {i--; continue;}

            else
              ;
          }
	
          fclose(fpo);
          printf("Traceback saved in result.txt\n");
        }


	printf("==============================================================\n");
	// OpenCL shutdown
	if(shutdown()) return -1;

	clReleaseMemObject(input_itemsets_d);
	clReleaseMemObject(output_itemsets_d);
	clReleaseMemObject(reference_d);

	free(reference);
	free(input_itemsets);
	free(output_itemsets);
	free(kernel_event1);
	free(kernel_event2);
	free(source);
}

