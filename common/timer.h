#ifndef COMMON_TIMER_H_
#define COMMON_TIMER_H_
#include <time.h>

//====================================================================================================================================
// Multi-platform general-puprpose high-precision timer
//====================================================================================================================================

// GetTime returns current time
// TimeDiff returns difference between start and end in ms
#ifdef _WIN32
	#include <Windows.h>
	
	#define TimeStamp _int64
	#define GetTime(X) QueryPerformanceCounter((LARGE_INTEGER*)&X)

	static inline double TimeDiff(_int64 start, _int64 end)
	{
		_int64 TimerFrequency;
		QueryPerformanceFrequency((LARGE_INTEGER*)&TimerFrequency);
		return (end - start) * 1000.0 / (double)(TimerFrequency);
	}
#elif __APPLE__
	#include <mach/mach_time.h>
	
	#define TimeStamp uint64_t
	#define GetTime(X) X=mach_absolute_time()
	
	static inline double TimeDiff(uint64_t start, uint64_t end)
	{
		mach_timebase_info_data_t info;
		mach_timebase_info(&info);
		
		return (double)((end - start) * (info.numer / info.denom)) / 1000000.0;
	}
#elif __linux
	#include <sys/time.h>
	
	#define TimeStamp struct timespec
	#define GetTime(X) clock_gettime(CLOCK_MONOTONIC_RAW, &X);
	
	static inline double TimeDiff(struct timespec start, struct timespec end)
	{
		return (double)( (end.tv_sec * 1000.0) + (end.tv_nsec/1000000.0) - (start.tv_sec * 1000.0) - (start.tv_nsec/1000000.0) );
	}
#else
	#error "Unsupported platform!"
#endif

#endif // COMMON_TIMER_H_
