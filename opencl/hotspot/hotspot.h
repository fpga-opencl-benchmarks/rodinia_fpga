#ifndef HOTSPOT_H
#define HOTSPOT_H

#include "OpenCL_helper_library.h"

#include "../common/opencl_util.h"
#include "hotspot_common.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>


#define STR_SIZE 256
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5


#define MIN(a, b) ((a)<=(b) ? (a) : (b))




/* chip parameters	*/
const static float t_chip = 0.0005;
const static float chip_height = 1.6;
const static float chip_width = 1.6;
/* ambient temperature, assuming no package at all	*/
const static float amb_temp = 80.0;

// OpenCL globals
cl_context context;
cl_command_queue command_queue;
cl_command_queue command_queue2;
cl_device_id device;
cl_kernel kernel;
cl_kernel ReadKernel;
cl_kernel WriteKernel;

#ifdef EMULATOR
cl_command_queue command_queue3;
cl_kernel ComputeKernel;
#endif

void writeoutput(float *, int, int, char *);
void readinput(float *, int, int, char *);
//int compute_tran_temp(cl_mem, cl_mem[2], int, int, int, int, int, int, int, int, float *, float *);
void usage(int, char **);
void run(int, char **);



#endif
