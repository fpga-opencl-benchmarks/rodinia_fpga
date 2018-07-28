// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	INCLUDE/DEFINE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>									// (in path known to compiler)	needed by printf
#include <stdlib.h>									// (in path known to compiler)	needed by malloc, free

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./main.h"									// (in current path)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./util/graphics/graphics.h"							// (in specified path)
#include "./util/graphics/resize.h"							// (in specified path)
//#include "./util/timer/timer.h"							// (in specified path)
#include "../../common/timer.h"
#include "../common/opencl_util.h"
#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
	#include "../../common/power_fpga.h"
#endif
//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel/kernel_gpu_opencl_wrapper.h"

//======================================================================================================================================================150
//	End
//======================================================================================================================================================150

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int main(int argc, char* argv [])
{
	int version;
	init_fpga(&argc, &argv, &version);
	printf("WG size of kernel = %d \n", NUMBER_THREADS);

	//======================================================================================================================================================150
	// 	VARIABLES
	//======================================================================================================================================================150

	// time
	double extractTime, computeTime, compressTime;
	TimeStamp time0;
	TimeStamp time1;
	TimeStamp time2;
	TimeStamp time3;
	TimeStamp time4;
	TimeStamp time5;
	TimeStamp time6;

#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
	// power parameters for Bittware A10PL4
	double power = 0;
	double energy = 0;
#endif

	// inputs image, input parameters
	fp* image_ori;													// original input image
	int image_ori_rows;
	int image_ori_cols;
	cl_long image_ori_elem;

	// inputs image, input parameters
	fp* image;													// input image
	int Nr,Nc;													// IMAGE nbr of rows/cols/elements
	cl_long Ne;

	// algorithm parameters
	int niter;													// nbr of iterations
	fp lambda;													// update step size

	// size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	cl_long NeROI;													// ROI nbr of elements

	// surrounding pixel indices
	int* iN = NULL;
	int* iS = NULL;
	int* jE = NULL;
	int* jW = NULL;    

	// counters
	cl_long i;    // image row
	cl_long j;    // image col

	// memory sizes
	int mem_size_i;
	int mem_size_j;

	GetTime(time0);

	//======================================================================================================================================================150
	//	INPUT ARGUMENTS
	//======================================================================================================================================================150

	if(argc != 5){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
	}

	GetTime(time1);

	//======================================================================================================================================================150
	// 	READ INPUT FROM FILE
	//======================================================================================================================================================150

	//====================================================================================================100
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//====================================================================================================100

	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

	read_graphics(	"../../data/srad/image.pgm",
					image_ori,
					image_ori_rows,
					image_ori_cols,
					1);

	//====================================================================================================100
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//====================================================================================================100

	Ne = Nr*Nc;

	image = (fp*)alignedMalloc(sizeof(fp) * Ne);

	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);

	//====================================================================================================100
	// 	End
	//====================================================================================================100

	GetTime(time2);

	//======================================================================================================================================================150
	// 	SETUP
	//======================================================================================================================================================150

	// variables
	r1     = 0;											// top row index of ROI
	r2     = Nr - 1;										// bottom row index of ROI
	c1     = 0;											// left column index of ROI
	c2     = Nc - 1;										// right column index of ROI

	// ROI image size
	NeROI = (r2-r1+1)*(c2-c1+1);									// number of elements in ROI, ROI size

	// allocate variables for surrounding pixels
	mem_size_i = sizeof(int) * Nr;
	mem_size_j = sizeof(int) * Nc;
	if (is_ndrange_kernel(version) || version < 5)
	{
		iN = (int *)alignedMalloc(mem_size_i) ;							// north surrounding element
		iS = (int *)alignedMalloc(mem_size_i) ;							// south surrounding element
		jW = (int *)alignedMalloc(mem_size_j) ;							// west surrounding element
		jE = (int *)alignedMalloc(mem_size_j) ;							// east surrounding element

		// N/S/W/E indices of surrounding pixels (every element of IMAGE)
		for (i=0; i<Nr; i++)
		{
			iN[i] = i-1;									// holds index of IMAGE row above
			iS[i] = i+1;									// holds index of IMAGE row below
		}
		for (j=0; j<Nc; j++)
		{
			jW[j] = j-1;									// holds index of IMAGE column on the left
			jE[j] = j+1;									// holds index of IMAGE column on the right
		}

		// N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
		iN[0]    = 0;										// changes IMAGE top row index from -1 to 0
		iS[Nr-1] = Nr-1;									// changes IMAGE bottom row index from Nr to Nr-1
		jW[0]    = 0;										// changes IMAGE leftmost column index from -1 to 0
		jE[Nc-1] = Nc-1;									// changes IMAGE rightmost column index from Nc to Nc-1
	}

	GetTime(time3);
	//======================================================================================================================================================150
	// 	KERNEL
	//======================================================================================================================================================150

	kernel_gpu_opencl_wrapper(	image,								// input image
                                        Nr,								// IMAGE nbr of rows
                                        Nc,								// IMAGE nbr of cols
                                        Ne,								// IMAGE nbr of elem
                                        niter,								// nbr of iterations
                                        lambda,								// update step size
                                        NeROI,								// ROI nbr of elements
                                        iN,
                                        iS,
                                        jE,
                                        jW,
                                        mem_size_i,
                                        mem_size_j,
                                        version,
//                                        &kernelRunTime,						// Kernel execution time
                                        &extractTime,							// Image extraction time
                                        &computeTime,							// Compute loop time
                                        &compressTime							// Image compression time
#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
                                     ,  &power								// Power usage for supported boards
#endif
	                         );
	GetTime(time4);

	//======================================================================================================================================================150
	// 	WRITE OUTPUT IMAGE TO FILE
	//======================================================================================================================================================150

	write_graphics(	"./output/image_out.pgm",
					image,
					Nr,
					Nc,
					1,
					255);

	GetTime(time5);

	//======================================================================================================================================================150
	// 	FREE MEMORY
	//======================================================================================================================================================150

	free(image_ori);
	free(image);
	free(iN); 
	free(iS); 
	free(jW); 
	free(jE);

	GetTime(time6);

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

        double total_ms = TimeDiff(time0, time5);
	printf("Time spent in different stages of the application:\n\n");
	printf("%.9f s, %.4f %% : READ COMMAND LINE PARAMETERS\n",
               TimeDiff(time0, time1) / 1000.0, TimeDiff(time0, time1) / total_ms * 100);
	printf("%.9f s, %.4f %% : READ AND RESIZE INPUT IMAGE FROM FILE\n",
               TimeDiff(time1, time2) / 1000.0, TimeDiff(time1, time2) / total_ms * 100);
	printf("%.9f s, %.4f %% : SETUP\n",
               TimeDiff(time2, time3) / 1000.0, TimeDiff(time2, time3) / total_ms * 100);
	// Below value accounts for anything that happens in kernel_gpu_opencl_wrapper other than image extraction, compute loop and image compression
	printf("%.9f s, %.4f %% : KERNEL PREPARATION\n",
               (TimeDiff(time3, time4) - extractTime - computeTime - compressTime) / 1000.0, (TimeDiff(time3, time4) - extractTime - computeTime - compressTime) / total_ms * 100);
	printf("%.9f s, %.4f %% : EXTRACT IMAGE\n",
               extractTime / 1000.0, extractTime / total_ms * 100);
	printf("%.9f s, %.4f %% : COMPUTE\n",
               computeTime / 1000.0, computeTime / total_ms * 100);	
	printf("%.9f s, %.4f %% : COMPRESS IMAGE\n",
               compressTime / 1000.0, compressTime / total_ms * 100);
	printf("%.9f s, %.4f %% : WRITE OUTPUT IMAGE TO FILE\n",
               TimeDiff(time4, time5) / 1000.0, TimeDiff(time4, time5) / total_ms * 100);
	printf("%.9f s, %.4f %% : FREE MEMORY\n",
               TimeDiff(time5, time6) / 1000.0, TimeDiff(time5, time6) / total_ms * 100);

	printf("\nComputation done in %0.3lf ms.\n", computeTime);
#ifdef AOCL_BOARD_a10pl4_dd4gb_gx115es3
	energy = GetEnergyFPGA(power, computeTime);
	if (power != -1) // -1 --> failed to read energy values
	{
		printf("Total energy used is %0.3lf jouls.\n", energy);
		printf("Average power consumption is %0.3lf watts.\n", power);
	}
#endif
	printf("Total time: %.3lf ms\n", total_ms);
	// The below value reflects only pure kernel execution time for all kernels in total
//	printf("Total kernel execution time (including extract and compress kernels): %.9f s\n", kernelRunTime / 1000.0);
        return 0;
}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200
