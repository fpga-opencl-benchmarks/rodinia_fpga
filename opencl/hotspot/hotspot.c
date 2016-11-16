#include "hotspot_common.h"
#include "hotspot.h"
#include "../../common/timer.h"
#include "../common/opencl_timer.h"

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j, index=0;
  FILE *fp;
  char str[STR_SIZE];

  if( (fp = fopen(file, "w" )) == 0 )
  {
    printf("could not open output file, skipping...\n");
  }
  else
  {
    for (i=0; i < grid_rows; i++)
    {
      for (j=0; j < grid_cols; j++)
      {
        sprintf(str, "%d\t%f\n", index, vect[i*grid_cols+j]);
        fputs(str,fp);
        index++;
      }
    }
    fclose(fp);
  }
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file) {

  int i,j;
  FILE *fp;
  char str[STR_SIZE];
  float val;

  printf("Reading %s\n", file);

  if( (fp  = fopen(file, "r" )) ==0 )
    fatal( "The input file was not opened" );


  for (i=0; i <= grid_rows-1; i++) 
    for (j=0; j <= grid_cols-1; j++)
    {
      if (fgets(str, STR_SIZE, fp) == NULL) fatal("Error reading file\n");
      if (feof(fp))
        fatal("not enough lines in file");
      //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
      if ((sscanf(str, "%f", &val) != 1))
        fatal("invalid file format");
      vect[i*grid_cols+j] = val;
    }

  fclose(fp);	

}


/*
  compute N time steps
*/

int compute_tran_temp(cl_mem MatrixPower, cl_mem MatrixTemp[2], int col, int row, \
                      int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows,
                      float *TempCPU, float *PowerCPU, int version_number,
                      int block_size) 
{
  float grid_height = chip_height / row;
  float grid_width = chip_width / col;

  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  float Rz = t_chip / (K_SI * grid_height * grid_width);

  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float step = PRECISION / max_slope;

  int src = 0, dst = 1;
  
  cl_mem boundary; // for v11
  
  if (version_number >= 11)
  {
    cl_int error;
    boundary = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * row * 4, NULL, &error);	// to store necessary boundary data for two blocks, two columns per block
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  }

  CL_SAFE_CALL(clFinish(command_queue));
  
  TimeStamp start_time;
  GetTime(start_time);

  if (is_ndrange_kernel(version_number))
  {
    // Determine GPU work group grid
    size_t global_work_size[2];
    global_work_size[0] = block_size * blockCols;
    global_work_size[1] = block_size * blockRows;
    size_t local_work_size[2];
    local_work_size[0] = block_size;
    local_work_size[1] = block_size;
    
    if (version_number == 4)
    {
      float step_div_Cap=step/Cap;
      float Rx_1=1/Rx;
      float Ry_1=1/Ry;
      float Rz_1=1/Rz;

      clSetKernelArg(kernel, 0 , sizeof(cl_mem), (void *) &MatrixPower);
      clSetKernelArg(kernel, 3 , sizeof(int)   , (void *) &col);
      clSetKernelArg(kernel, 4 , sizeof(int)   , (void *) &row);
      clSetKernelArg(kernel, 5 , sizeof(float) , (void *) &step_div_Cap);
      clSetKernelArg(kernel, 6 , sizeof(float) , (void *) &Rx_1);
      clSetKernelArg(kernel, 7 , sizeof(float) , (void *) &Ry_1);
      clSetKernelArg(kernel, 8 , sizeof(float) , (void *) &Rz_1);
    }
    else
    {
      clSetKernelArg(kernel, 1 , sizeof(cl_mem), (void *) &MatrixPower);
      clSetKernelArg(kernel, 4 , sizeof(int)   , (void *) &col);
      clSetKernelArg(kernel, 5 , sizeof(int)   , (void *) &row);
      clSetKernelArg(kernel, 6 , sizeof(int)   , (void *) &borderCols);
      clSetKernelArg(kernel, 7 , sizeof(int)   , (void *) &borderRows);
      clSetKernelArg(kernel, 8 , sizeof(float) , (void *) &Cap);
      clSetKernelArg(kernel, 9 , sizeof(float) , (void *) &Rx);
      clSetKernelArg(kernel, 10, sizeof(float) , (void *) &Ry);
      clSetKernelArg(kernel, 11, sizeof(float) , (void *) &Rz);
      clSetKernelArg(kernel, 12, sizeof(float) , (void *) &step);
    }

    int t;
    for (t = 0; t < total_iterations; t += num_iterations)
    {
      int iter = MIN(num_iterations, total_iterations - t);

      if (version_number == 4)
      {
        clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &MatrixTemp[src]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[dst]);
      }
      else
      {
        clSetKernelArg(kernel, 0, sizeof(int), (void *) &iter);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[src]);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &MatrixTemp[dst]);
      }

      // Launch kernel
      cl_int error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
      if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

      src = 1 - src;
      dst = 1 - dst;
    }
  }
  else
  {
    // Using the single work-item versions.
    // All iterations are computed by a single kernel execution.
    // borderCols and borderRows are not used in these versions, so
    // they are not passed.
    if (version_number >= 11)
    {
      float step_div_Cap=step/Cap;
      float Rx_1=1/Rx;
      float Ry_1=1/Ry;
      float Rz_1=1/Rz;

      clSetKernelArg(kernel_even, 0, sizeof(cl_mem), (void *) &MatrixPower);
      clSetKernelArg(kernel_even, 2, sizeof(int)   , (void *) &col);
      clSetKernelArg(kernel_even, 3, sizeof(int)   , (void *) &row);
      clSetKernelArg(kernel_even, 4, sizeof(float) , (void *) &step_div_Cap);
      clSetKernelArg(kernel_even, 5, sizeof(float) , (void *) &Rx_1);
      clSetKernelArg(kernel_even, 6, sizeof(float) , (void *) &Ry_1);
      clSetKernelArg(kernel_even, 7, sizeof(float) , (void *) &Rz_1);
      clSetKernelArg(kernel_even, 8, sizeof(cl_mem), (void *) &boundary);

      clSetKernelArg(kernel_odd , 0, sizeof(cl_mem), (void *) &MatrixPower);
      clSetKernelArg(kernel_odd , 2, sizeof(int)   , (void *) &col);
      clSetKernelArg(kernel_odd , 3, sizeof(int)   , (void *) &row);
      clSetKernelArg(kernel_odd , 4, sizeof(float) , (void *) &step_div_Cap);
      clSetKernelArg(kernel_odd , 5, sizeof(float) , (void *) &Rx_1);
      clSetKernelArg(kernel_odd , 6, sizeof(float) , (void *) &Ry_1);
      clSetKernelArg(kernel_odd , 7, sizeof(float) , (void *) &Rz_1);
      clSetKernelArg(kernel_odd , 8, sizeof(cl_mem), (void *) &boundary);
    }
    else
    {
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &MatrixPower);
      clSetKernelArg(kernel, 3, sizeof(int),    (void *) &col);
      clSetKernelArg(kernel, 4, sizeof(int),    (void *) &row);
      clSetKernelArg(kernel, 5, sizeof(float),  (void *) &Cap);
      clSetKernelArg(kernel, 6, sizeof(float),  (void *) &Rx);
      clSetKernelArg(kernel, 7, sizeof(float),  (void *) &Ry);
      clSetKernelArg(kernel, 8, sizeof(float),  (void *) &Rz);
      clSetKernelArg(kernel, 9, sizeof(float),  (void *) &step);
    }

    // Launch kernel
    int t;
    if (version_number >= 11)
    {
      for (t = 0; t < total_iterations; t += 2) // in this version two iterations are launched simultaneously, hence total iteration number is halved
      {
        int rem_iter = total_iterations-t;
        clSetKernelArg(kernel_even, 1, sizeof(cl_mem), (void *) &MatrixTemp[src]);
        clSetKernelArg(kernel_odd , 1, sizeof(cl_mem), (void *) &MatrixTemp[dst]);

	// pass number of remaining iterations to correctly handle cases where number of iterations is an odd number
	clSetKernelArg(kernel_even, 9, sizeof(int)   , (void *) &rem_iter);
	clSetKernelArg(kernel_odd , 9, sizeof(int)   , (void *) &rem_iter);

        CL_SAFE_CALL(clEnqueueTask(command_queue , kernel_even, 0, NULL, NULL));
        CL_SAFE_CALL(clEnqueueTask(command_queue2, kernel_odd , 0, NULL, NULL));

        clFinish(command_queue2); // this is necessary since the two kernels are running in two different queue and a new iteration should not start before the previous one finishes

        src = 1 - src;
        dst = 1 - dst;
      }
    }
    else
    {
      for (t = 0; t < total_iterations; ++t)
      {
        clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &MatrixTemp[src]);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[dst]);
        CL_SAFE_CALL(clEnqueueTask(command_queue, kernel, 0, NULL, NULL));
        src = 1 - src;
        dst = 1 - dst;
      }
    }
  }

  // Wait for all operations to finish
  CL_SAFE_CALL(clFinish(command_queue));

  TimeStamp end_time;
  GetTime(end_time);
  printf("\nComputation done in %0.3lf ms.\n", TimeDiff(start_time, end_time));
  
  if (version_number >= 11)
  {
    clReleaseMemObject(boundary);
  }

  return src;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file> <block_size>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file\n");
  fprintf(stderr, "\t<block_size> - kernel block size (optional)\n");  
  exit(1);
}

int main(int argc, char** argv) {
  char *version_string;  
  int version_number;
  init_fpga2(&argc, &argv, &version_string, &version_number);  

  size_t devices_size;
  cl_int result, error;
  cl_uint platformCount;
  cl_platform_id* platforms = NULL;
  cl_context_properties ctxprop[3];
  cl_device_type   device_type;

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
  result = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size );
  int num_devices = (int) (devices_size / sizeof(cl_device_id));

  if( result != CL_SUCCESS || num_devices < 1 )
  {
    printf("ERROR: clGetContextInfo() failed\n");
    return -1;
  }
  cl_device_id *device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
  if( !device_list )
  {
    printf("ERROR: new cl_device_id[] failed\n");
    return -1;
  }
  CL_SAFE_CALL(clGetContextInfo( context, CL_CONTEXT_DEVICES, devices_size, device_list, NULL ));
  device = device_list[0];

  // Create command queue
  command_queue = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, &error );
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  if (version_number >= 11)
  {
    command_queue2 = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, &error );
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  }

  int size;
  int grid_rows,grid_cols = 0;
  float *FilesavingTemp,*FilesavingPower; //,*MatrixOut; 
  char *tfile, *pfile, *ofile;
  int block_size = 16; // default if not specified at the command line
    
  int total_iterations = 60;
  int pyramid_height = 1; // number of iterations

  if (argc < 7)
    usage(argc, argv);
  if((grid_rows = atoi(argv[1]))<=0||
     (grid_cols = atoi(argv[1]))<=0||
     (pyramid_height = atoi(argv[2]))<=0||
     (total_iterations = atoi(argv[3]))<0)
    usage(argc, argv);

  // pyramid_height must be 1 as the temporal blocking is not supported. It is
  // implemented in the original OpenCL kernel, but removed in our versions,
  // including the v0 kernel, for simplifying initial performance evaluations.
  // It also reduces logic usage.
  if (pyramid_height != 1) {
      fprintf(stderr, "Error! pyramid_height parameter must be 1 as the temporal blocking is not supported.\n");
      exit(1);
  }

  tfile=argv[4];
  pfile=argv[5];
  ofile=argv[6];
  if (argc >= 8) 
    block_size = atoi(argv[7]);

  size=grid_rows*grid_cols;

  // --------------- pyramid parameters --------------- 
  int borderCols = (pyramid_height)*EXPAND_RATE/2;
  int borderRows = (pyramid_height)*EXPAND_RATE/2;
  int smallBlockCol = block_size-(pyramid_height)*EXPAND_RATE;
  int smallBlockRow = block_size-(pyramid_height)*EXPAND_RATE;
  int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
  int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

  FilesavingTemp = (float *) alignedMalloc(size*sizeof(float));
  FilesavingPower = (float *) alignedMalloc(size*sizeof(float));
  // MatrixOut = (float *) calloc (size, sizeof(float));

  if( !FilesavingPower || !FilesavingTemp) // || !MatrixOut)
    fatal("unable to allocate memory");

  // Read input data from disk
  readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
  readinput(FilesavingPower, grid_rows, grid_cols, pfile);

  // Load kernel source from file
  char *kernel_file_path = getVersionedKernelName2("hotspot_kernel", version_string);

  size_t sourceSize;  
  char *source = read_kernel(kernel_file_path, &sourceSize);

#ifdef USE_JIT
  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &error);
#else
  cl_program program = clCreateProgramWithBinary(context, 1, device_list, &sourceSize, (const unsigned char**)&source, NULL, &error);
#endif
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

  char clOptions[110];
  sprintf(clOptions, "-I.");
#if defined(USE_JIT)
#ifdef USE_RESTRICT
  sprintf(clOptions + strlen(clOptions), " -DUSE_RESTRICT");
#endif  
  sprintf(clOptions + strlen(clOptions), " -DBSIZE=%d", block_size);
#endif

  // Create an executable from the kernel
  clBuildProgram_SAFE(program, 1, &device, clOptions, NULL, NULL);

  // Create kernel
  if (version_number >= 11)
  {
    kernel_even = clCreateKernel(program, "hotspot_even", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    kernel_odd = clCreateKernel(program, "hotspot_odd", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);    
  }
  else
  {
    kernel = clCreateKernel(program, "hotspot", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  }

  TimeStamp start_time;
  GetTime(start_time);

  // Create two temperature matrices and copy the temperature input data
  cl_mem MatrixTemp[2];
  // Create input memory buffers on device
  MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

  CL_SAFE_CALL(clEnqueueWriteBuffer(command_queue, MatrixTemp[0], CL_TRUE, 0, sizeof(float) * size, FilesavingTemp, 0, NULL, NULL));

  // Copy the power input data
  cl_mem MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

  CL_SAFE_CALL(clEnqueueWriteBuffer(command_queue, MatrixPower, CL_TRUE, 0, sizeof(float) * size, FilesavingPower, 0, NULL, NULL));

  // Perform the computation
  int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height,
                              blockCols, blockRows, borderCols, borderRows, FilesavingTemp, FilesavingPower,
                              version_number, block_size);

  // Copy final temperature data back
  float *MatrixOut = (float*)alignedMalloc(sizeof(float) * size);
  CL_SAFE_CALL(clEnqueueReadBuffer(command_queue, MatrixTemp[ret], CL_TRUE, 0, sizeof(float) * size, MatrixOut, 0, NULL, NULL));

  TimeStamp end_time;
  GetTime(end_time);
  printf("Total run time was %f ms.\n", TimeDiff(start_time, end_time));

  // Write final output to output file
  writeoutput(MatrixOut, grid_rows, grid_cols, ofile);

  clReleaseMemObject(MatrixTemp[0]);
  clReleaseMemObject(MatrixTemp[1]);
  clReleaseMemObject(MatrixPower);

  clReleaseContext(context);

  return 0;
}
