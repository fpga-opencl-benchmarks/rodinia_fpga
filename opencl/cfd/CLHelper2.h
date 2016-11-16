#ifndef CLHELPER2_H_
#define CLHELPER2_H_

#include "../common/opencl_util.h"

// local variables
extern cl_context       context;
extern cl_command_queue cmd_queue;
//static cl_command_queue cmd_queue2;
extern cl_device_type   device_type;
extern cl_device_id   * device_list;
extern cl_int           num_devices;

static inline int _clInit() {
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

  free(platforms); // platforms isn't needed in the main function

  return 0;
}

static inline cl_mem _clMalloc(int size) {
  cl_int err;
  cl_mem m = clCreateBuffer(
      context, CL_MEM_READ_WRITE, size, NULL, &err);
  if (err != CL_SUCCESS) {
    display_error_message(err, stderr);
    exit(1);
  }
  return m;
}

static inline void _clMemcpyH2D(cl_mem dst, const void *src, int size) {
  CL_SAFE_CALL(clEnqueueWriteBuffer(
      cmd_queue, dst, CL_TRUE, 0, size, src, 0, NULL, NULL));
}

static inline void _clMemcpyD2D(cl_mem dst, cl_mem src, int size) {
  CL_SAFE_CALL(clEnqueueCopyBuffer(
      cmd_queue, src, dst, 0, 0, size, 0, NULL, NULL));
}

static inline void _clMemcpyD2H(void * dst, cl_mem src, int size) {
  CL_SAFE_CALL(clEnqueueReadBuffer(
      cmd_queue, src, CL_TRUE, 0, size, dst, 0, NULL, NULL));
}

static inline void _clFree(cl_mem ob) {
  CL_SAFE_CALL(clReleaseMemObject(ob));
}

static inline void _clFinish() {
  CL_SAFE_CALL(clFinish(cmd_queue));
}

static inline void _clRelease() {
  if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
  if( context ) clReleaseContext( context );
  if( device_list ) delete device_list;

  // reset all variables
  cmd_queue = 0;
  context = 0;
  device_list = 0;
  num_devices = 0;
  device_type = 0;
  return;
}

static inline void _clSetArgs(cl_kernel kernel, int arg_idx, void * d_mem, int size = 0) {
  if(!size){
    CL_SAFE_CALL(clSetKernelArg(kernel, arg_idx, sizeof(d_mem), &d_mem));
  } else {
    CL_SAFE_CALL(clSetKernelArg(kernel, arg_idx, size, d_mem));
  }
}

static inline void _clInvokeKernel(cl_kernel kernel, size_t work_items, size_t work_group_size) {
  cl_uint work_dim = 2;
  //process situations that work_items cannot be divided by work_group_size
  if(work_items%work_group_size != 0) {
    //work_items = work_items +
    //(work_group_size-(work_items%work_group_size));
    abort();
  }
  size_t local_work_size[] = {work_group_size, 1};
  size_t global_work_size[] = {work_items, 1};
  cl_event e[1];  
  CL_SAFE_CALL(clEnqueueNDRangeKernel(
      cmd_queue, kernel, work_dim, 0,
      global_work_size, local_work_size, 0 , 0, &(e[0])));
}

static inline void _clInvokeKernel(cl_kernel kernel) {
  CL_SAFE_CALL(clEnqueueTask(cmd_queue, kernel, 0, NULL, NULL));
}

static inline void _clStatistics() {
}
static inline void _clMemset(cl_mem mem_d, short val, int number_bytes) {
  // clEnqueueFillBuffer is available since OpenCL 1.2, which may not
  // bre supported by NVIDIA 
#if 0  
  CL_SAFE_CALL(clEnqueueFillBuffer(cmd_queue, mem_d, &val, sizeof(short),
                                   0, number_bytes, 0, NULL, NULL));
#else
  short *buf = (short *)malloc(number_bytes);
  for (int i = 0; i < number_bytes / (int)sizeof(short); ++i) {
    buf[i] = val;
  }
  _clMemcpyH2D(mem_d, buf, number_bytes);
  free(buf);
#endif
}
#endif /* CLHELPER2_H_ */
