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

#include "../common/opencl_util.h"
#include "../../common/timer.h"

using namespace std;

#define HALO     1 // halo width along one direction when advancing to the next iteration
#define BLOCK_SIZE 256
#define STR_SIZE 256
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

void init(int argc, char** argv)
{
	if (argc == 4)
	{
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
		pyramid_height = atoi(argv[3]);
	}
	else
	{
		printf("Usage: dynproc row_len col_len pyramid_height\n");
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
	
	resultFile = fopen("result.txt", "w");
	if (resultFile == NULL)
	{
		printf("Failed to open result file!\n");
		exit(-1);
	}
	init_fpga(&argc, &argv, &version);
	init(argc, argv);
	
	// Pyramid parameters.
	int borderCols = (pyramid_height) * HALO;
	int smallBlockCol = BLOCK_SIZE - (pyramid_height) * HALO * 2;
	int blockCols = cols / smallBlockCol + ((cols % smallBlockCol == 0) ? 0 : 1);

	
	fprintf(resultFile, "pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
		pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

	int size = rows * cols;

	// Create and initialize the OpenCL object.
	OpenCL cl(1);  // 1 means to display output (debugging mode).
	cl.init(version);
	cl.gwSize(BLOCK_SIZE*blockCols);
	cl.lwSize(BLOCK_SIZE);

	// Create and build the kernel.
	string kn = "dynproc_kernel";  // the kernel name, for future use.
	cl.createKernel(kn);

	// Allocate device memory.
	cl_mem d_gpuWall;
	cl_mem d_gpuResult[2];
	if (is_ndrange_kernel(version))
	{
		d_gpuWall = clCreateBuffer(cl.ctxt(),
						CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						sizeof(cl_int)*(size-cols),
						(data + cols),
						NULL);

		d_gpuResult[0] = clCreateBuffer(cl.ctxt(),
						CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
						sizeof(cl_int)*cols,
						data,
						NULL);
	}
	else if (version != 5 && version != 7 && version != 9) 
	{
		d_gpuWall = clCreateBuffer(cl.ctxt(),
						CL_MEM_READ_ONLY,
						sizeof(cl_int)*(size-cols),
						NULL,
						NULL);

		d_gpuResult[0] = clCreateBuffer(cl.ctxt(),
						CL_MEM_READ_WRITE,
						sizeof(cl_int)*cols,
						NULL,
						NULL);
        }
        else
        {
		d_gpuWall = clCreateBuffer(cl.ctxt(),
						CL_MEM_READ_ONLY,
						sizeof(cl_int)*(size),
						NULL,
						NULL);
	}


	d_gpuResult[1] = clCreateBuffer(cl.ctxt(),
					CL_MEM_READ_WRITE,
					sizeof(cl_int)*cols,
					NULL,
					NULL);

	int src, dst;
	if (is_ndrange_kernel(version))
	{
		src = 1;
		dst = 0;
		
		// Set fixed kernel arguments.
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 0, sizeof(cl_int), (void*) &pyramid_height));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 1, sizeof(cl_mem), (void*) &d_gpuWall));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 4, sizeof(cl_int), (void*) &cols));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 5, sizeof(cl_int), (void*) &rows));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 7, sizeof(cl_int), (void*) &borderCols));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 8, sizeof(cl_int) * BLOCK_SIZE, 0));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 9, sizeof(cl_int) * BLOCK_SIZE, 0));
		
		GetTime(kernelStart);
		for (int startStep = 0; startStep < rows - 1; startStep += pyramid_height)
		{
			int temp = src;
			src = dst;
			dst = temp;

			// Calculate changed kernel arguments...
			CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 2, sizeof(cl_mem), (void*) &d_gpuResult[src]));
			CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 3, sizeof(cl_mem), (void*) &d_gpuResult[dst]));
			CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 6, sizeof(cl_int), (void*) &startStep));
			
			// Launch kernel
			cl.launch(kn, version);
		}
	}
	else if (version != 5 && version != 7 && version != 9) 
	{
		src = 0;
		dst = 1;
		
		// Set kernel arguments.
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 0, sizeof(cl_mem), (void*) &d_gpuWall));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 1, sizeof(cl_mem), (void*) &d_gpuResult[src]));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 2, sizeof(cl_mem), (void*) &d_gpuResult[dst]));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 3, sizeof(cl_int), (void*) &cols));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 4, sizeof(cl_int), (void*) &rows));
		
		CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuWall, CL_TRUE, 0, sizeof(cl_int)*(size-cols), (data + cols), 0, NULL, NULL));
		CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuResult[0], CL_TRUE, 0, sizeof(cl_int)*cols, (data), 0, NULL, NULL));
		
		GetTime(kernelStart);
		// Launch kernel
		cl.launch(kn, version);
	}
        else
        {
                dst = 1;
                CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 0, sizeof(cl_mem), (void*) &d_gpuWall));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 1, sizeof(cl_mem), (void*) &d_gpuResult[dst]));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 2, sizeof(cl_int), (void*) &cols));
		CL_SAFE_CALL(clSetKernelArg(cl.kernel(kn), 3, sizeof(cl_int), (void*) &rows));
          
          	CL_SAFE_CALL(clEnqueueWriteBuffer(cl.command_queue, d_gpuWall, CL_TRUE, 0, sizeof(cl_int)*(size), data, 0, NULL, NULL));

                GetTime(kernelStart);
                cl.launch(kn, version);
        }
	clFinish(cl.command_queue);
	GetTime(kernelEnd);

	// Copy results back to host.
	clEnqueueReadBuffer(cl.q(),                   // The command queue.
	                    d_gpuResult[dst],         // The result on the device.
	                    CL_TRUE,                  // Blocking? (ie. Wait at this line until read has finished?)
	                    0,                        // Offset. None in this case.
	                    sizeof(cl_int)*cols,      // Size to copy.
	                    result,                   // The pointer to the memory on the host.
	                    0,                        // Number of events in wait list. Not used.
	                    NULL,                     // Event wait list. Not used.
	                    NULL);                    // Event object for determining status. Not used.
	
#ifdef BENCH_PRINT

	for (int i = 0; i < cols; i++)
	{
		fprintf(resultFile, "%d ",data[i]) ;
	}
	fprintf(resultFile, "\n") ;

#endif

	for (int i = 0; i < cols; i++)
	{
		fprintf(resultFile, "%d ",result[i]) ;
	}
	fprintf(resultFile, "\n") ;
	
	printf("Kernel execution time is: %0.6lf ms\n", TimeDiff(kernelStart, kernelEnd));

	// Memory cleanup here.
	free(data);
	free(wall);
	free(result);
	fclose(resultFile);
	
	return EXIT_SUCCESS;
}
