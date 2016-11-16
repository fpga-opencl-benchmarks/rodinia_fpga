#ifndef COMMON_OPENCL_TIMER_H_
#define COMMON_OPENCL_TIMER_H_
#include <time.h>
#include "../common/opencl_util.h"

#define START 0
#define END   1

//====================================================================================================================================
// Helper functions for OpenCL's built-in timer --> only for kernel execution time calculation
//====================================================================================================================================

// CLGetTime returns current time
// CLTimeDiff returns difference between start and end in ms, valid options are START and END
#define CLTimeStamp cl_ulong
static inline cl_ulong CLGetTime(cl_event event, unsigned short option)
{
	CLTimeStamp time;
	if ( option == START )
	{
		CL_SAFE_CALL( clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time, NULL) );
	}
	else if ( option == END )
	{
		CL_SAFE_CALL( clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time, NULL) );
	}
	else
	{
		printf("Invalid CLGetTime option!\n");
		exit(-1);
	}
	
	return (time);
}

static inline double CLTimeDiff(cl_ulong start, cl_ulong end)
{
	return (double)( (end - start) / 1000000.0 );
}

#endif // COMMON_TIMER_H_
