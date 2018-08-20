/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include "OpenCL.h"
#include "pathfinder_common.h"

#include "../common/opencl_util.h"
#include "../../common/timer.h"
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
	#include "../../common/power_fpga.h"
#endif

using namespace std;

#define HALO     1 // halo width along one direction when advancing to the next iteration
#define DEVICE   0
#define M_SEED   9
//#define BENCH_PRINT
#define IN_RANGE(x, min, max)	((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

// Program variables.
int   rows, cols;
int   Ne = rows * cols;
int*  data;
int** wall;
int*  result;
int   pyramid_height;
FILE *resultFile;
char* ofile = NULL;
bool write_out = 0;

void init(int argc, char** argv)
{
	if (argc == 4 || argc == 5)
	{
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
		pyramid_height = atoi(argv[3]);
		if (argc == 5)
		{
			ofile = argv[4];
			write_out = 1;
		}			
	}
	else
	{
		printf("Usage: %s row_len col_len pyramid_height output_file\n", argv[0]);
		exit(0);
	}
	data = (int *)alignedMalloc( rows * cols * sizeof(int) );
	wall = (int **)alignedMalloc( rows * sizeof(int*) );
	for (int n = 0; n < rows; n++)
	{
		// wall[n] is set to be the nth row of the data array.
		wall[n] = data + cols * n;
	}
	result = (int *)alignedMalloc( cols * sizeof(int) );

	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			wall[i][j] = rand() % 10;
		}
	}
	
#ifdef BENCH_PRINT

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			fprintf(resultFile, "%d ", wall[i][j]);
		}
		fprintf(resultFile, "\n");
	}

#endif
}

void fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);
}

int main(int argc, char** argv)
{
	int version;
	TimeStamp kernelStart, kernelEnd;
	double computeTime;

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
	// power measurement parameters, only for Bittware and Nallatech's Arria 10 boards
	int flag = 0;
	double power = 0;
	double energy = 0;
#endif

	init_fpga(&argc, &argv, &version);
	init(argc, argv);

	if (write_out)
	{
		resultFile = fopen(ofile, "w");
		if (resultFile == NULL)
		{
			printf("Failed to open result file!\n");
			exit(-1);
		}
	}
	
	// Pyramid parameters.
	int borderCols = (pyramid_height) * HALO;
	int smallBlockCol = BSIZE - (pyramid_height) * HALO * 2;
	int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

	
	fprintf(stdout, "pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
		pyramid_height, cols, borderCols, BSIZE, blockCols, smallBlockCol);

	int size = rows * cols;

	// Create and initialize the OpenCL object.
	OpenCL cl(1);  // 1 means to display output (debugging mode).
	cl.init(version);
	cl.gwSize(BSIZE*blockCols);
	cl.lwSize(BSIZE);

	// Create and build the kernel.
	string kn = "dynproc_kernel";  // the kernel name, for future use.
	cl.createKernel(kn);

	// Allocate device memory.
	cl_mem d_gpuWall;
	cl_mem d_gpuResult[2];
	cl_int error;
	if (is_ndrange_kernel(version) || version == 5)
	{
		d_gpuWall      = clCreateBuffer(cl.ctxt(), CL_MEM_READ_ONLY , sizeof(cl_int)*(size-cols), NULL, &error);
		if (error != CL_SUCCESS){ printf("Failed to allocate device buffer!\n"); exit(-1); }
		d_gpuResult[0] = clCreateBuffer(cl.ctxt(), CL_MEM_READ_WRITE, sizeof(cl_int)*cols       , NULL, &error);
		if (error != CL_SUCCESS){ printf("Failed to allocate device buffer!\n"); exit(-1); }

		CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuWall     , CL_TRUE, 0, sizeof(cl_int)*(size-cols), data + cols, 0, NULL, NULL));
		CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuResult[0], CL_TRUE, 0, sizeof(cl_int)*cols       , data       , 0, NULL, NULL));
	}
	else if (version <= 3)
	{
		d_gpuWall      = clCreateBuffer(cl.ctxt(), CL_MEM_READ_ONLY , sizeof(cl_int)*size, NULL, &error);
		if (error != CL_SUCCESS){ printf("Failed to allocate device buffer!\n"); exit(-1); }
		d_gpuResult[0] = clCreateBuffer(cl.ctxt(), CL_MEM_READ_WRITE, sizeof(cl_int)*cols, NULL, &error);
		if (error != CL_SUCCESS){ printf("Failed to allocate device buffer!\n"); exit(-1); }

		CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuWall     , CL_TRUE, 0, sizeof(cl_int)*(size), data, 0, NULL, NULL));
		CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuResult[0], CL_TRUE, 0, sizeof(cl_int)*cols  , data, 0, NULL, NULL));
	}

	d_gpuResult[1] = clCreateBuffer(cl.ctxt(), CL_MEM_READ_WRITE, sizeof(cl_int)*cols, NULL, &error);
	if (error != CL_SUCCESS){ printf("Failed to allocate device buffer!\n"); exit(-1); }

	int src, dst;
#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
	#pragma omp parallel num_threads(2) shared(flag)
	{
		if (omp_get_thread_num() == 0)
		{
			#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115
				power = GetPowerFPGA(&flag);
			#else
				power = GetPowerFPGA(&flag, cl.devices);
			#endif
		}
		else
		{
			#pragma omp barrier
#endif
			if (is_ndrange_kernel(version))
			{
				src = 1;
				dst = 0;

				// Set fixed kernel arguments
				CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 1, sizeof(cl_mem), (void*) &d_gpuWall));
				CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 4, sizeof(cl_int), (void*) &cols));
				CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 6, sizeof(cl_int), (void*) &borderCols));

				GetTime(kernelStart);
				for (int startStep = 0; startStep < rows - 1; startStep += pyramid_height)
				{
					int temp = src;
					src = dst;
					dst = temp;

					// Calculate changed kernel arguments...
					int iteration = MIN(pyramid_height, rows - startStep - 1);
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 0, sizeof(cl_int), (void*) &iteration));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 2, sizeof(cl_mem), (void*) &d_gpuResult[src]));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 3, sizeof(cl_mem), (void*) &d_gpuResult[dst]));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 5, sizeof(cl_int), (void*) &startStep));
					
					// Launch kernel
					cl.launch(kn, version);

					clFinish(cl.command_queue);
				}
			}
			else if (version <= 3)
			{
				src = 1;
				dst = 0;

				// Set fixed kernel arguments
				CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 0, sizeof(cl_mem), (void*) &d_gpuWall));
				CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 3, sizeof(cl_int), (void*) &cols));

				GetTime(kernelStart);
				for (int t = 0; t < rows - 1; t++)
				{
					int temp = src;
					src = dst;
					dst = temp;

					// Calculate changed kernel arguments...
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 1, sizeof(cl_mem), (void*) &d_gpuResult[src]));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 2, sizeof(cl_mem), (void*) &d_gpuResult[dst]));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 4, sizeof(cl_int), (void*) &t));
					
					// Launch kernel
					cl.launch(kn, version);

					clFinish(cl.command_queue);
				}
			}
			else
			{
				src = 1;
				dst = 0;

				// Set kernel arguments
				CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 0, sizeof(cl_mem), (void*) &d_gpuWall));
				CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 3, sizeof(cl_int), (void*) &cols));

				GetTime(kernelStart);

				for (int startStep = 0; startStep < rows - 1; startStep += pyramid_height)
				{
					int temp = src;
					src = dst;
					dst = temp;

					int rem_rows = MIN(pyramid_height, rows - startStep - 1);	// either equal to pyramid_height or the number of remaining iterations

					// Exit condition should be a multiple of comp_bsize
					int comp_bsize = BSIZE - 2 * rem_rows;
					int last_col   = (cols % comp_bsize == 0) ? cols + 0 : cols + comp_bsize - cols % comp_bsize;
					int col_blocks = last_col / comp_bsize;
					int comp_exit  = BSIZE * col_blocks * (rem_rows + 1) / SSIZE;

					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 1, sizeof(cl_mem), (void*) &d_gpuResult[src]));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 2, sizeof(cl_mem), (void*) &d_gpuResult[dst]));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 4, sizeof(cl_int), (void*) &rem_rows));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 5, sizeof(cl_int), (void*) &startStep));
					CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 6, sizeof(cl_int), (void*) &comp_exit));

					// Launch kernel
					cl.launch(kn, version);

					clFinish(cl.command_queue);
				}
			}

			clFinish(cl.command_queue);
			GetTime(kernelEnd);

#if defined(AOCL_BOARD_a10pl4_dd4gb_gx115) || defined(AOCL_BOARD_p385a_sch_ax115)
			flag = 1;
		}
	}
#endif

	// Copy results back to host.
	clEnqueueReadBuffer(cl.q(), d_gpuResult[dst], CL_TRUE, 0, sizeof(cl_int)*cols, result, 0, NULL, NULL);

	if (write_out)
	{
		#ifdef BENCH_PRINT
		for (int i = 0; i < cols; i++)
		{
			fprintf(resultFile, "%d ", data[i]) ;
		}
		fprintf(resultFile, "\n") ;
		#endif

		for (int i = 0; i < cols; i++)
		{
			fprintf(resultFile, "%d\n", result[i]) ;
		}
		fclose(resultFile);
	}

	computeTime = TimeDiff(kernelStart, kernelEnd);
	printf("\nComputation done in %0.3lf ms.\n", computeTime);

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

	// Memory cleanup here.
	free(data);
	free(wall);
	free(result);

	return EXIT_SUCCESS;
}
