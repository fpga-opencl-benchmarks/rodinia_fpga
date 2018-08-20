#include "hotspot.h"
#include "../../common/timer.h"

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
     #include "../../common/power_fpga.h"
#endif

cl_device_id *device_list;

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
                      int total_iterations, int pyramid_height, int blockCols, int blockRows, int haloCols, int haloRows,
                      int version_number, int block_size_x, int block_size_y) 
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

  TimeStamp compute_start, compute_end;
  double computeTime;
  
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  // power measurement parameters, only for Bittware and Nallatech's Arria 10 boards
  int flag = 0;
  double power = 0;
  double energy = 0;
#endif

  CL_SAFE_CALL(clFinish(command_queue));

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  #pragma omp parallel num_threads(2) shared(flag)
  {
    if (omp_get_thread_num() == 0)
    {
      #ifdef AOCL_BOARD_a10pl4_dd4gb_gx115
        power = GetPowerFPGA(&flag);
      #else
        power = GetPowerFPGA(&flag, device_list);
      #endif
    }
    else
    {
      #pragma omp barrier
#endif
      if (is_ndrange_kernel(version_number))
      {
        // Determine GPU work group grid
        size_t global_work_size[2];
        global_work_size[0] = block_size_x * blockCols;
        global_work_size[1] = block_size_y * blockRows;
        size_t local_work_size[2];
        local_work_size[0] = block_size_x;
        local_work_size[1] = block_size_y;

        if (version_number == 2)
        {
          float step_div_Cap=step/Cap;
          float Rx_1=1/Rx;
          float Ry_1=1/Ry;
          float Rz_1=1/Rz;

          CL_SAFE_CALL(clSetKernelArg(kernel, 1 , sizeof(cl_mem), (void *) &MatrixPower ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 4 , sizeof(int)   , (void *) &col         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 5 , sizeof(int)   , (void *) &row         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 6 , sizeof(int)   , (void *) &haloCols  ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 7 , sizeof(int)   , (void *) &haloRows  ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 8 , sizeof(float) , (void *) &step_div_Cap));
          CL_SAFE_CALL(clSetKernelArg(kernel, 9 , sizeof(float) , (void *) &Rx_1        ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 10, sizeof(float) , (void *) &Ry_1        ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 11, sizeof(float) , (void *) &Rz_1        ));
        }
        else if (version_number == 4)
        {
          float step_div_Cap=step/Cap;
          float Rx_1=1/Rx;
          float Ry_1=1/Ry;
          float Rz_1=1/Rz;

          CL_SAFE_CALL(clSetKernelArg(kernel, 1 , sizeof(cl_mem), (void *) &MatrixPower    ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 4 , sizeof(int)   , (void *) &col            ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 5 , sizeof(int)   , (void *) &row            ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 6 , sizeof(int)   , (void *) &pyramid_height ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 7 , sizeof(float) , (void *) &step_div_Cap   ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 8 , sizeof(float) , (void *) &Rx_1           ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 9 , sizeof(float) , (void *) &Ry_1           ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 10, sizeof(float) , (void *) &Rz_1           ));
        }
        else
        {
          CL_SAFE_CALL(clSetKernelArg(kernel, 1 , sizeof(cl_mem), (void *) &MatrixPower ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 4 , sizeof(int)   , (void *) &col         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 5 , sizeof(int)   , (void *) &row         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 6 , sizeof(int)   , (void *) &haloCols    ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 7 , sizeof(int)   , (void *) &haloRows    ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 8 , sizeof(float) , (void *) &Cap         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 9 , sizeof(float) , (void *) &Rx          ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 10, sizeof(float) , (void *) &Ry          ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 11, sizeof(float) , (void *) &Rz          ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 12, sizeof(float) , (void *) &step        ));
        }

        // Beginning of timing point
        GetTime(compute_start);

        // Launch kernel
        int t;
        for (t = 0; t < total_iterations; t += pyramid_height)
        {
          int iter = MIN(pyramid_height, total_iterations - t);
          
          // each block finally computes result for a small block
          // after N iterations. 
          // it is the non-overlapping small blocks that cover 
          // all the input data

          // calculate the small block size
          int small_block_cols = BLOCK_X - iter * 2; //EXPAND_RATE
          int small_block_rows = BLOCK_Y - iter * 2; //EXPAND_RATE

          CL_SAFE_CALL(clSetKernelArg(kernel, 0 , sizeof(int)   , (void *) &iter           ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 2 , sizeof(cl_mem), (void *) &MatrixTemp[src]));
          CL_SAFE_CALL(clSetKernelArg(kernel, 3 , sizeof(cl_mem), (void *) &MatrixTemp[dst]));
          if (version_number == 2)
          {
               CL_SAFE_CALL(clSetKernelArg(kernel, 12, sizeof(int), (void *) &small_block_rows));
               CL_SAFE_CALL(clSetKernelArg(kernel, 13, sizeof(int), (void *) &small_block_cols));
          }
          else if (version_number == 4)
          {
               CL_SAFE_CALL(clSetKernelArg(kernel, 11, sizeof(int), (void *) &small_block_rows));
               CL_SAFE_CALL(clSetKernelArg(kernel, 12, sizeof(int), (void *) &small_block_cols));
          }

          // Launch kernel
          CL_SAFE_CALL( clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL) );

          clFinish(command_queue);

          src = 1 - src;
          dst = 1 - dst;
        }
      }
      else
      {
        // Using the single work-item versions.
        // All iterations are computed by a single kernel execution.
        // haloCols and haloRows are not used in these versions, so
        // they are not passed.
        if (version_number >= 7)
        {
          // Exit condition should be a multiple of comp_bsize
          int comp_bsize = BLOCK_X - BACK_OFF;
          int last_col   = (col % comp_bsize == 0) ? col + 0 : col + comp_bsize - col % comp_bsize;
          int col_blocks = last_col / comp_bsize;
          int comp_exit  = BLOCK_X * col_blocks * (row + 1) / SSIZE;
          int mem_exit   = BLOCK_X * col_blocks * row / SSIZE;

          float step_div_Cap=step/Cap;
          float Rx_1=1/Rx;
          float Ry_1=1/Ry;
          float Rz_1=1/Rz;

          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 1, sizeof(cl_mem), (void *) &MatrixPower ));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 2, sizeof(int)   , (void *) &col         ));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 3, sizeof(int)   , (void *) &row         ));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 4, sizeof(float) , (void *) &step_div_Cap));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 5, sizeof(float) , (void *) &Rx_1        ));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 6, sizeof(float) , (void *) &Ry_1        ));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 7, sizeof(float) , (void *) &Rz_1        ));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 8, sizeof(float) , (void *) &comp_exit   ));
          CL_SAFE_CALL(clSetKernelArg(ReadKernel, 9, sizeof(float) , (void *) &mem_exit    ));

          CL_SAFE_CALL(clSetKernelArg(WriteKernel, 1, sizeof(int)  , (void *) &col         ));
          CL_SAFE_CALL(clSetKernelArg(WriteKernel, 2, sizeof(int)  , (void *) &row         ));
          CL_SAFE_CALL(clSetKernelArg(WriteKernel, 3, sizeof(float), (void *) &mem_exit    ));
        }
        else if (version_number == 5)
        {
          // Exit condition should be a multiple of comp_bsize
          int comp_bsize = BLOCK_X - 2;
          int last_col   = (col % comp_bsize == 0) ? col + 0 : col + comp_bsize - col % comp_bsize;
          int col_blocks = last_col / comp_bsize;
          int comp_exit  = BLOCK_X * col_blocks * (row + 1) / SSIZE;

          float step_div_Cap=step/Cap;
          float Rx_1=1/Rx;
          float Ry_1=1/Ry;
          float Rz_1=1/Rz;

          CL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &MatrixPower ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(int)   , (void *) &col         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 4, sizeof(int)   , (void *) &row         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 5, sizeof(float) , (void *) &step_div_Cap));
          CL_SAFE_CALL(clSetKernelArg(kernel, 6, sizeof(float) , (void *) &Rx_1        ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 7, sizeof(float) , (void *) &Ry_1        ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 8, sizeof(float) , (void *) &Rz_1        ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 9, sizeof(float) , (void *) &comp_exit   ));
        }
        else if (version_number == 3)
        {
          float step_div_Cap=step/Cap;
          float Rx_1=1/Rx;
          float Ry_1=1/Ry;
          float Rz_1=1/Rz;

          CL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &MatrixPower ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(int)   , (void *) &col         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 4, sizeof(int)   , (void *) &row         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 5, sizeof(float) , (void *) &step_div_Cap));
          CL_SAFE_CALL(clSetKernelArg(kernel, 6, sizeof(float) , (void *) &Rx_1        ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 7, sizeof(float) , (void *) &Ry_1        ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 8, sizeof(float) , (void *) &Rz_1        ));
        }
        else
        {
          CL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &MatrixPower ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(int)   , (void *) &col         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 4, sizeof(int)   , (void *) &row         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 5, sizeof(float) , (void *) &Cap         ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 6, sizeof(float) , (void *) &Rx          ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 7, sizeof(float) , (void *) &Ry          ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 8, sizeof(float) , (void *) &Rz          ));
          CL_SAFE_CALL(clSetKernelArg(kernel, 9, sizeof(float) , (void *) &step        ));
        }

        // Beginning of timing point
        GetTime(compute_start);

        // Launch kernel
        int t;
        if (version_number >= 7)
        {
          for (t = 0; t < total_iterations; t = t + TIME) // in this version two iterations are launched simultaneously, hence total iteration number is halved
          {
            CL_SAFE_CALL(clSetKernelArg(ReadKernel , 0, sizeof(cl_mem), (void *) &MatrixTemp[src]));
            CL_SAFE_CALL(clSetKernelArg(WriteKernel, 0, sizeof(cl_mem), (void *) &MatrixTemp[dst]));

            // the following variable is used to choose which time step should send its output to the write kernel
            // unless there are less iterations left than there are parallel time steps, the last time step will do so
            int rem_iter = (total_iterations - t > TIME) ? TIME : total_iterations - t;
            CL_SAFE_CALL(clSetKernelArg(ReadKernel, 10, sizeof(float) , (void *) &rem_iter));

            CL_SAFE_CALL(clEnqueueTask(command_queue , ReadKernel   , 0, NULL, NULL));

            CL_SAFE_CALL(clEnqueueTask(command_queue2, WriteKernel  , 0, NULL, NULL));
#ifdef EMULATOR
            CL_SAFE_CALL(clEnqueueTask(command_queue3, ComputeKernel, 0, NULL, NULL));
#endif

            clFinish(command_queue2); // this is necessary since the two kernels are running in two different queue and a new iteration should not start before the previous one finishes

            src = 1 - src;
            dst = 1 - dst;
          }
        }
        else
        {
          for (t = 0; t < total_iterations; ++t)
          {
            CL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &MatrixTemp[src]));
            CL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &MatrixTemp[dst]));
            CL_SAFE_CALL(clEnqueueTask(command_queue, kernel, 0, NULL, NULL));

            clFinish(command_queue);

            src = 1 - src;
            dst = 1 - dst;
          }
        }
      }

      // Wait for all operations to finish
      clFinish(command_queue);

      GetTime(compute_end);
      
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
      flag = 1;
    }
  }
#endif

  computeTime = TimeDiff(compute_start, compute_end);
  printf("\nComputation done in %0.3lf ms.\n", computeTime);
  printf("Throughput is %0.3lf GBps.\n", (3 * row * col * sizeof(float) * total_iterations) / (1000000000.0 * computeTime / 1000.0));

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  energy = GetEnergyFPGA(power, computeTime);
  if (power != -1) // -1 --> sensor read failure
  {
    printf("Total energy used is %0.3lf jouls.\n", energy);
    printf("Average power consumption is %0.3lf watts.\n", power);
  }
  else
  {
    printf("Failed to read power values from the sensor!\n");
  }
#endif

  return src;
}

void usage(int argc, char **argv) {
  fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
  fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<pyramid_height> - pyramid height(positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  fprintf(stderr, "\t<output_file> - name of the output file (optional)\n");
  
  fprintf(stderr, "\tNote: If output file name is not supplied, output will not be written to disk.\n");
  exit(1);
}

int main(int argc, char** argv) {
  int write_out = 0;
  char *version_string;  
  int version_number;
  init_fpga2(&argc, &argv, &version_string, &version_number);

  int size;
  int grid_rows,grid_cols = 0;
  float *FilesavingTemp = NULL,*FilesavingPower = NULL;
  char *tfile, *pfile, *ofile = NULL;
  int block_size_x = BLOCK_X;
  int block_size_y = BLOCK_Y;
    
  int total_iterations = 60;
  int pyramid_height = 1; // number of combined iterations

  if (argc < 5)
    usage(argc, argv);
  if((grid_rows = atoi(argv[1]))<=0||
     (grid_cols = atoi(argv[1]))<=0||
     (pyramid_height = atoi(argv[2]))<=0||
     (total_iterations = atoi(argv[3]))<0)
    usage(argc, argv);

  size_t devices_size;
  cl_int result, error;
  cl_uint platformCount;
  cl_platform_id* platforms = NULL;
  cl_context_properties ctxprop[3];
  cl_device_type   device_type;

  display_device_info(&platforms, &platformCount);
  select_device_type(platforms, &platformCount, &device_type);
  validate_selection(platforms, &platformCount, ctxprop, &device_type);
  
  TimeStamp total_start, total_end;

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
  device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
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
  if (version_number >= 7)
  {
    command_queue2 = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, &error );
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  }

#ifdef EMULATOR
  if (version_number >= 7)
  {
    command_queue3 = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, &error );
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  }
#endif

  tfile=argv[4];
  pfile=argv[5];
  if (argc >= 7)
  {
    write_out = 1;
    ofile=argv[6];
  }

  size = (version_number < 7) ? grid_rows * grid_cols : grid_rows * grid_cols + PAD;

  // --------------- pyramid parameters --------------- 
  int haloCols = (pyramid_height) * EXPAND_RATE / 2;
  int haloRows = (pyramid_height) * EXPAND_RATE / 2;
  int smallBlockCol = block_size_x - (pyramid_height) * EXPAND_RATE;
  int smallBlockRow = block_size_y - (pyramid_height) * EXPAND_RATE;
  int blockCols = grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
  int blockRows = grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);

  FilesavingTemp  = (float *) alignedMalloc(size * sizeof(float));
  FilesavingPower = (float *) alignedMalloc(size * sizeof(float));

  if (!FilesavingPower || !FilesavingTemp)
    fatal("unable to allocate memory");

  // Read input data from disk
  if (version_number < 7)
  {
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);
  }
  else
  {
    readinput(FilesavingTemp + PAD, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower + PAD, grid_rows, grid_cols, pfile);
  }

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
  if (version_number == 4)
  {
    sprintf(clOptions + strlen(clOptions), " -DBLOCK_X=%d -DBLOCK_Y=%d -DSSIZE=%d", block_size_x, block_size_y, SSIZE);
  }
  else
  {
    sprintf(clOptions + strlen(clOptions), " -DBSIZE=%d -DSSIZE=%d", block_size_x, SSIZE);
  }
#endif

  // Create an executable from the kernel
  clBuildProgram_SAFE(program, 1, &device, clOptions, NULL, NULL);

  // Create kernel
  if (version_number >= 7)
  {
    ReadKernel = clCreateKernel(program, "read", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
    WriteKernel = clCreateKernel(program, "write", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
#ifdef EMULATOR
    ComputeKernel = clCreateKernel(program, "compute", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__); 
#endif
  }
  else
  {
    kernel = clCreateKernel(program, "hotspot", &error);
    if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  }

  GetTime(total_start);

  // Create two temperature matrices and copy the temperature input data
  cl_mem MatrixTemp[2], MatrixPower = NULL;

  // Create input memory buffers on device
  MatrixTemp[0] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);
  MatrixTemp[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

  CL_SAFE_CALL(clEnqueueWriteBuffer(command_queue, MatrixTemp[0], CL_TRUE, 0, sizeof(float) * size, FilesavingTemp, 0, NULL, NULL));

  // Copy the power input data
  MatrixPower = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * size, NULL, &error);
  if (error != CL_SUCCESS) fatal_CL(error, __LINE__);

  CL_SAFE_CALL(clEnqueueWriteBuffer(command_queue, MatrixPower, CL_TRUE, 0, sizeof(float) * size, FilesavingPower, 0, NULL, NULL));

  // Perform the computation
  int ret = compute_tran_temp(MatrixPower, MatrixTemp, grid_cols, grid_rows, total_iterations, pyramid_height,
                              blockCols, blockRows, haloCols, haloRows, version_number, block_size_x, block_size_y);

  // Copy final temperature data back
  float *MatrixOut = NULL;

  MatrixOut = (float*)alignedMalloc(sizeof(float) * size);
  CL_SAFE_CALL(clEnqueueReadBuffer(command_queue, MatrixTemp[ret], CL_TRUE, 0, sizeof(float) * size, MatrixOut, 0, NULL, NULL));

  GetTime(total_end);
  printf("Total run time was %f ms.\n", TimeDiff(total_start, total_end));

  // Write final output to output file
  if (write_out)
  {
    if (version_number < 7)
    {
      writeoutput(MatrixOut, grid_rows, grid_cols, ofile);
    }
    else
    {
      writeoutput(MatrixOut + PAD, grid_rows, grid_cols, ofile);
    }
  }

  clReleaseMemObject(MatrixTemp[0]);
  clReleaseMemObject(MatrixTemp[1]);
  clReleaseMemObject(MatrixPower);

  clReleaseContext(context);

  return 0;
}
