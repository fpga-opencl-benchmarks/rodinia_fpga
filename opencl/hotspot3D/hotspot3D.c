#include "hotspot3D_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <CL/cl.h>
#include "CL_helper.h"
#include "../common/opencl_util.h"
#include "../../common/timer.h"
#include <stdlib.h>

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
	#include "../../common/power_fpga.h"
#endif

#ifndef DEVICE
#define DEVICE CL_DEVICE_TYPE_DEFAULT
#endif

#define TOL      (0.001)
#define STR_SIZE (256)
#define MAX_PD   (3.0e6)

/* required precision in degrees	*/
#define PRECISION    0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI         100

/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5

float t_chip      = 0.0005;
float chip_height = 0.016;
float chip_width  = 0.016;
float amb_temp    = 80.0;

void usage(int argc, char **argv)
{
  fprintf(stderr, "Usage: %s <rows> <cols> <layers> <iterations> <powerFile> <tempFile> <outputFile>\n", argv[0]);
  fprintf(stderr, "\t<rows>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<cols>  - number of rows/cols in the grid (positive integer)\n");
  fprintf(stderr, "\t<layers>     - number of layers in the grid (positive integer)\n");
  fprintf(stderr, "\t<iteration>  - number of iterations\n");
  fprintf(stderr, "\t<powerFile>  - name of the file containing the initial power values of each cell\n");
  fprintf(stderr, "\t<tempFile>   - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr, "\t<outputFile> - output file\n\n");
  
  fprintf(stderr, "\tNote: If input file names are not supplied, input is generated randomly.\n");
  fprintf(stderr, "\tNote: If output file name is not supplied, output will not be written to disk.\n");
  exit(1);
}



int main(int argc, char** argv)
{
  int write_out = 0;
  int version;
  TimeStamp start, end;
  
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
  // power measurement parameters, only for Bittware and Nallatech's Arria 10 boards
  int flag = 0;
  double power = 0;
  double energy = 0;
#endif

  init_fpga(&argc, &argv, &version);

  if (argc < 5 || argc > 8)
  {
    usage(argc,argv);
  }

  char *pfile = NULL, *tfile = NULL, *ofile = NULL;
  int iterations = atoi(argv[4]);

  if (argc == 8)
  {
    write_out      = 1;
    pfile          = argv[5];
    tfile          = argv[6];
    ofile          = argv[7];
  }
  else if (argc == 7)
  {
    pfile          = argv[5];
    tfile          = argv[6];
  }
  else if (argc == 6)
  {
    write_out      = 1;
    ofile          = argv[5];
  }

  int numCols      = atoi(argv[1]);
  int numRows      = atoi(argv[2]);
  int layers       = atoi(argv[3]);

  /* calculating parameters*/

  float dx         = chip_height/numRows;
  float dy         = chip_width/numCols;
  float dz         = t_chip/layers;

  float Cap        = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * dx * dy;
  float Rx         = dy / (2.0 * K_SI * t_chip * dx);
  float Ry         = dx / (2.0 * K_SI * t_chip * dy);
  float Rz         = dz / (K_SI * dx * dy);

  float max_slope  = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  float dt         = PRECISION / max_slope;

  float ce, cw, cn, cs, ct, cb, cc;
  float stepDivCap = dt / Cap;
  ce               = cw                                              = stepDivCap/ Rx;
  cn               = cs                                              = stepDivCap/ Ry;
  ct               = cb                                              = stepDivCap/ Rz;

  cc               = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

  cl_int          err;
  size_t          devices_size;
  int size = (version >= 7) ? numCols * numRows * layers + PAD : numCols * numRows * layers;
  float*        tIn      = (float*)alignedCalloc(size, sizeof(float));
  float*        pIn      = (float*)alignedCalloc(size, sizeof(float));
  float*        tempCopy = (float*)alignedMalloc(size * sizeof(float));
  float*        tempOut  = (float*)alignedCalloc(size, sizeof(float));
  int count = size;
  if (argc == 7)
  {
    if (version >= 7)
    {
      readinput(tIn + PAD, numRows, numCols, layers, tfile);
      readinput(pIn + PAD, numRows, numCols, layers, pfile);
    }
    else
    {
      readinput(tIn, numRows, numCols, layers, tfile);
      readinput(pIn, numRows, numCols, layers, pfile);
    }
  }
  else
  {
    srand(10);
    int i;
    if (version >= 7)
    {
      for (i = PAD; i < size; i++)
      {
        pIn[i] = (float)rand() / (float)(RAND_MAX); // random number between 0 and 1
        tIn[i] = 300 + (float)rand() / (float)(RAND_MAX/100); // random number between 300 and 400
      }
    }
    else
    {
      for (i = 0; i < size; i++)
      {
        pIn[i] = (float)rand() / (float)(RAND_MAX); // random number between 0 and 1
        tIn[i] = 300 + (float)rand() / (float)(RAND_MAX/100); // random number between 300 and 400
      }
    }
  }

  size_t global[2];                   
  size_t local[2];
  memcpy(tempCopy, tIn, size * sizeof(float));

  cl_context       context;
  cl_command_queue command_queue;
  cl_command_queue command_queue2 = NULL;
  cl_program       program;
  cl_kernel        hotspot3D = NULL;
  cl_kernel        ReadKernel = NULL;
  cl_kernel        WriteKernel = NULL;
  cl_device_type   device_type;

#ifdef EMULATOR
  cl_command_queue command_queue3 = NULL;
  cl_kernel        ComputeKernel = NULL;
#endif

  cl_mem d_a;                     
  cl_mem d_b;                     
  cl_mem d_c;                     

  //const char *KernelSource = load_kernel_source("hotspotKernel.cl");
  size_t sourcesize;
  char *kernel_file_path = getVersionedKernelName("./hotspot3D_kernel", version);
  char *KernelSource = read_kernel(kernel_file_path, &sourcesize);
  free(kernel_file_path);

  /*cl_uint numPlatforms;

  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to find a platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  cl_platform_id Platform[numPlatforms];
  err = clGetPlatformIDs(numPlatforms, Platform, NULL);
  if (err != CL_SUCCESS || numPlatforms <= 0)
    {
      printf("Error: Failed to get the platform!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  for (i = 0; i < numPlatforms; i++)
    {
      err = clGetDeviceIDs(Platform[i], DEVICE, 1, &device_id, NULL);
      if (err == CL_SUCCESS)
        {
          break;
        } 
    }

  if (device_id == NULL)
    {
      printf("Error: Failed to create a device group!\n%s\n",err_code(err));
      return EXIT_FAILURE;
    }

  err = output_device_info(device_id);*/

  cl_platform_id *platforms = NULL;
  cl_uint num_platforms = 0;
  cl_context_properties ctxprop[3];
  display_device_info(&platforms, &num_platforms);
  select_device_type(platforms, &num_platforms, &device_type);
  validate_selection(platforms, &num_platforms, ctxprop, &device_type);
  
  context = clCreateContextFromType( ctxprop, device_type, NULL, NULL, &err );
  if ( err != CL_SUCCESS )
    {
      printf("Error: Failed to create context from type!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  err = clGetContextInfo( context, CL_CONTEXT_DEVICES, 0, NULL, &devices_size );
  int num_devices = (int) (devices_size / sizeof(cl_device_id));
  if( err != CL_SUCCESS || num_devices < 1 )
    {
      printf("Error: Failed to get context info!\n");
      return -1;
    }

  cl_device_id *device_list = (cl_device_id*)malloc(sizeof(cl_device_id)*num_devices);
  if( !device_list )
    {
      printf("Error: Failed to create device list!\n");
      return -1;
    }

  CL_SAFE_CALL(clGetContextInfo( context, CL_CONTEXT_DEVICES, devices_size, device_list, NULL ));

  /*context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
    {
      printf("Error: Failed to create a compute context!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }*/

  command_queue = clCreateCommandQueue(context, device_list[0], 0, &err);
  if (!command_queue)
  {
    printf("Error: Failed to create command_queue!\n%s\n", err_code(err));
    return EXIT_FAILURE;
  }
  if (version >= 7)
  {
    command_queue2 = clCreateCommandQueue(context, device_list[0], 0, &err);
    if (!command_queue2)
    {
      printf("Error: Failed to create a command queue2!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }
#ifdef EMULATOR
    command_queue3 = clCreateCommandQueue(context, device_list[0], 0, &err);
    if (!command_queue3)
    {
      printf("Error: Failed to create a command queue3!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }
#endif
  }

#if defined(USE_JIT)
  program = clCreateProgramWithSource(context, 1, (const char **) &KernelSource, NULL, &err);
#else
  program = clCreateProgramWithBinary(context, 1, device_list, &sourcesize, (const unsigned char**)&KernelSource, NULL, &err);
#endif
  if (!program)
    {
      printf("Error: Failed to create compute program!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

  clBuildProgram_SAFE(program, num_devices, device_list, NULL, NULL, NULL);

  if (version >= 7)
  {
    ReadKernel = clCreateKernel(program, "read", &err);
    if (!ReadKernel || err != CL_SUCCESS)
    {
      printf("Error: Failed to create read kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }

    WriteKernel = clCreateKernel(program, "write", &err);
    if (!WriteKernel || err != CL_SUCCESS)
    {
      printf("Error: Failed to create write kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }
#ifdef EMULATOR
    ComputeKernel = clCreateKernel(program, "compute", &err);
    if (!ComputeKernel || err != CL_SUCCESS)
    {
      printf("Error: Failed to create compute kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }
#endif
  }
  else
  {
    hotspot3D = clCreateKernel(program, "hotspotOpt1", &err);
    if (!hotspot3D || err != CL_SUCCESS)
    {
      printf("Error: Failed to create hotspot3D kernel!\n%s\n", err_code(err));
      return EXIT_FAILURE;
    }
  }

  d_a  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);
  d_b  = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(float) * count, NULL, NULL);
  d_c  = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * count, NULL, NULL);

  if (!d_a || !d_b || !d_c) 
    {
      printf("Error: Failed to allocate device memory!\n");
      exit(1);
    }    

  err = clEnqueueWriteBuffer(command_queue, d_a, CL_TRUE, 0, sizeof(float) * count, tIn, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write tIn to source array!\n%s\n", err_code(err));
      exit(1);
    }

  err = clEnqueueWriteBuffer(command_queue, d_b, CL_TRUE, 0, sizeof(float) * count, pIn, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write pIn to source array!\n%s\n", err_code(err));
      exit(1);
    }

  err = clEnqueueWriteBuffer(command_queue, d_c, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to write tempOut to source array!\n%s\n", err_code(err));
      exit(1);
    }
    
  // fixed arguments
  if (version >= 7)
  {
    // Exit condition should be a multiple of comp_bsize_{x|y}
    int comp_bsize_x = BLOCK_X - BACK_OFF;
    int comp_bsize_y = BLOCK_Y - BACK_OFF;
    int last_col     = ((numCols % comp_bsize_x == 0) ? numCols + 0 : numCols + comp_bsize_x - numCols % comp_bsize_x) - comp_bsize_x; // exit variable is first compared, then incremented
    int last_row     = ((numRows % comp_bsize_y == 0) ? numRows + 0 : numRows + comp_bsize_y - numRows % comp_bsize_y) - comp_bsize_y; // exit variable is first compared, then incremented
    int col_blocks   = (last_col / comp_bsize_x) + 1;
    int row_blocks   = (last_row / comp_bsize_y) + 1;
    int comp_exit    = ((BLOCK_X * col_blocks * BLOCK_Y * row_blocks * (layers + RAD)) / SSIZE); // exit variable is first incremented, then compared
    int mem_exit     = (BLOCK_X * col_blocks * BLOCK_Y * row_blocks * layers) / SSIZE;           // exit variable is first incremented, then compared

    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 0 , sizeof(cl_mem), &d_b       ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 2 , sizeof(float) , &stepDivCap));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 3 , sizeof(int)   , &numCols   ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 4 , sizeof(int)   , &numRows   ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 5 , sizeof(int)   , &layers    ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 6 , sizeof(float) , &ce        ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 7 , sizeof(float) , &cw        ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 8 , sizeof(float) , &cn        ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 9 , sizeof(float) , &cs        ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 10, sizeof(float) , &ct        ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 11, sizeof(float) , &cb        ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 12, sizeof(float) , &cc        ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 13, sizeof(int)   , &last_col  ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 14, sizeof(int)   , &comp_exit ));
    CL_SAFE_CALL(clSetKernelArg(ReadKernel, 15, sizeof(int)   , &mem_exit  ));

    CL_SAFE_CALL(clSetKernelArg(WriteKernel, 1, sizeof(int)   , &numCols   ));
    CL_SAFE_CALL(clSetKernelArg(WriteKernel, 2, sizeof(int)   , &numRows   ));
    CL_SAFE_CALL(clSetKernelArg(WriteKernel, 3, sizeof(int)   , &layers    ));
    CL_SAFE_CALL(clSetKernelArg(WriteKernel, 4, sizeof(int)   , &last_col  ));
    CL_SAFE_CALL(clSetKernelArg(WriteKernel, 5, sizeof(int)   , &mem_exit  ));
  }
  else if (version == 5)
  {
    // Exit condition should be a multiple of comp_bsize_{x|y}
    int comp_bsize_x = BLOCK_X - 2;
    int comp_bsize_y = BLOCK_Y - 2;
    int last_col     = ((numCols % comp_bsize_x == 0) ? numCols + 0 : numCols + comp_bsize_x - numCols % comp_bsize_x) - comp_bsize_x; // exit variable is first compared, then incremented
    int last_row     = ((numRows % comp_bsize_y == 0) ? numRows + 0 : numRows + comp_bsize_y - numRows % comp_bsize_y) - comp_bsize_y; // exit variable is first compared, then incremented
    int col_blocks   = (last_col / comp_bsize_x) + 1;
    int row_blocks   = (last_row / comp_bsize_y) + 1;
    int comp_exit    = ((BLOCK_X * col_blocks * BLOCK_Y * row_blocks * (layers + 1)) / SSIZE);   // exit variable is first incremented, then compared

    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 0 , sizeof(cl_mem), &d_b       ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 3 , sizeof(float) , &stepDivCap));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 4 , sizeof(int)   , &numCols   ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 5 , sizeof(int)   , &numRows   ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 6 , sizeof(int)   , &layers    ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 7 , sizeof(float) , &ce        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 8 , sizeof(float) , &cw        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 9 , sizeof(float) , &cn        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 10, sizeof(float) , &cs        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 11, sizeof(float) , &ct        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 12, sizeof(float) , &cb        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 13, sizeof(float) , &cc        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 14, sizeof(int)   , &last_col  ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 15, sizeof(int)   , &comp_exit ));
  }
  else
  {
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 0 , sizeof(cl_mem), &d_b       ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 3 , sizeof(float) , &stepDivCap));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 4 , sizeof(int)   , &numCols   ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 5 , sizeof(int)   , &numRows   ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 6 , sizeof(int)   , &layers    ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 7 , sizeof(float) , &ce        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 8 , sizeof(float) , &cw        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 9 , sizeof(float) , &cn        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 10, sizeof(float) , &cs        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 11, sizeof(float) , &ct        ));
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 12, sizeof(float) , &cb        ));      
    CL_SAFE_CALL(clSetKernelArg(hotspot3D, 13, sizeof(float) , &cc        ));
  }

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
    GetTime(start);
    int j;
    if (version >= 7)
    {
      for(j = 0; j < iterations; j += TIME)
      {
        int rem_iter = (iterations - j > TIME) ? TIME : iterations - j;
	   CL_SAFE_CALL(clSetKernelArg(ReadKernel , 16, sizeof(int)   , &rem_iter));

        CL_SAFE_CALL(clSetKernelArg(ReadKernel , 1 , sizeof(cl_mem), &d_a     ));
        CL_SAFE_CALL(clSetKernelArg(WriteKernel, 0 , sizeof(cl_mem), &d_c     ));

        CL_SAFE_CALL(clEnqueueTask(command_queue,  ReadKernel   , 0, NULL, NULL));
        CL_SAFE_CALL(clEnqueueTask(command_queue2, WriteKernel  , 0, NULL, NULL));
#ifdef EMULATOR
        CL_SAFE_CALL(clEnqueueTask(command_queue3, ComputeKernel, 0, NULL, NULL));
#endif
        clFinish(command_queue2);  // wait only for write kernel

        cl_mem temp = d_a;
        d_a         = d_c;
        d_c         = temp;
      }
    }
    else
    {
      for(j = 0; j < iterations; j++)
      {
        CL_SAFE_CALL(clSetKernelArg(hotspot3D, 1,  sizeof(cl_mem), &d_a));
        CL_SAFE_CALL(clSetKernelArg(hotspot3D, 2,  sizeof(cl_mem), &d_c));

        if (is_ndrange_kernel(version))
        {
          global[0] = numCols;
          global[1] = numRows;

          local[0] = WG_SIZE_X;
          local[1] = WG_SIZE_Y;

          CL_SAFE_CALL( clEnqueueNDRangeKernel(command_queue, hotspot3D, 2, NULL, global, local, 0, NULL, NULL) );
        }
        else
        {
          CL_SAFE_CALL( clEnqueueTask(command_queue, hotspot3D, 0, NULL, NULL) );
        }

	   clFinish(command_queue);

        cl_mem temp = d_a;
        d_a         = d_c;
        d_c         = temp;
      }
    }

    GetTime(end);

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
      flag = 1;
    }
  }
#endif

  // pointers are always swapped one extra time at the end of the iteration loop and hence, d_a points to the output, not d_c
  err = clEnqueueReadBuffer( command_queue, d_a, CL_TRUE, 0, sizeof(float) * count, tempOut, 0, NULL, NULL );
  if (err != CL_SUCCESS)
    {
      printf("Error: Failed to read output array!\n%s\n", err_code(err));
      exit(1);
    }

  double computeTime = TimeDiff(start, end);
  printf("Computation done in %0.3lf ms.\n", computeTime);
  printf("Throughput is %0.3lf GBps.\n", (3 * numCols * numRows * layers * sizeof(float) * iterations) / (1000000000.0 * computeTime / 1000.0));
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

#ifdef VERIFY
  float* answer = (float*)calloc(size, sizeof(float));
  if (version >= 7)
  {
    computeTempCPU(pIn + PAD, tempCopy + PAD, answer + PAD, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);
  }
  else
  {
    computeTempCPU(pIn, tempCopy, answer, numCols, numRows, layers, Cap, Rx, Ry, Rz, dt, amb_temp, iterations);
  }

  // for an even number of iterations, "tempCopy" will point to correct output of the CPU function
  // and for an odd number, "answer" will
  float* CPUOut = (iterations % 2 == 1) ? answer : tempCopy;
  float acc = (version >= 7) ? accuracy(tempOut + PAD, CPUOut + PAD, numRows * numCols * layers) : accuracy(tempOut, CPUOut, numRows * numCols * layers);

  printf("Accuracy: %e\n",acc);
#endif

  if (write_out)
  {
    if (version >= 7)
    {
      writeoutput(tempOut + PAD, numRows, numCols, layers, ofile);
    }
    else
    {
      writeoutput(tempOut, numRows, numCols, layers, ofile);
    }
  }

  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(hotspot3D);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);

  return 0;
}
