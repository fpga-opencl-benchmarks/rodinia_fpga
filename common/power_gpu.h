#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>
#include <nvml.h>

//====================================================================================================================================
// CUDA-based (NVML) GPU Energy Calculator
//====================================================================================================================================

// Nvidia's NVML library and header are needed
// Example compilation command is as follows:
// nvcc -Xcompiler -fopenmp test.cu -o test -L/usr/local/nvidia_gdk/usr/src/gdk/nvml/lib -I/usr/local/nvidia_gdk/usr/include/nvidia/gdk -lnvidia-ml

// Returns average power usage in Watt from when it is called until when "flag" becomes one
// DeviceID is used to choose target GPU in multi-GPU configurations
// Sampling is done every 10 milliseconds
// The host code should have two OpenMP threads, one running the CUDA kernel and the other calling this function
// A "#pragma omp barrier" should be put before the kernel call
// Flag should become one in the kernel thread after kernel execution has finished (after cudaThreadSynchronize())
static inline double GetPowerGPU(int* flag, int DeviceID)
{
	nvmlDevice_t device;
	nvmlReturn_t error;
	unsigned int power;
	size_t count = 0, powerSum = 0;

	// Initialize NVML library
	error = nvmlInit();  
	if (error != NVML_SUCCESS)
	{
		printf("Failed to initialize NVML API with error code \"%s\".\n", nvmlErrorString(error));
		#pragma omp barrier
		return -1;
	}

	// Get device handle
	error = nvmlDeviceGetHandleByIndex(DeviceID , &device);
	if (error != NVML_SUCCESS)
	{
		printf("Failed to get device handle with error code \"%s\".\n", nvmlErrorString(error));
		#pragma omp barrier
		return -1;
	}

	#pragma omp barrier
	while(*flag == 0)
	{
		// Returns device power usage in mWatt
		error = nvmlDeviceGetPowerUsage(device, &power);
		if(error != NVML_SUCCESS)
		{
			printf("Failed to get device power usage with error code \"%s\".\n", nvmlErrorString(error));
			return -1;
		}
		powerSum = powerSum + power;
		count++;
		
		// Sleep for 10 ms
		usleep(10000);
	}
        
	error = nvmlShutdown();
	if (error != NVML_SUCCESS)
	{
		printf("Failed to shutdown NVML API with error code \"%s\".\n", nvmlErrorString(error));
		return -1;
	}

	return (double)(powerSum)/(double)(count * 1000.0); // Wattage is in mWatt, hence the division by 1000
}

// Returns amount of energy used in jouls
// "power" is average power usage in Watt from the GetPowerGPU() fucntion
// "time" is run time in ms from one of our time measurement helper functions
static inline double GetEnergyGPU(double power, double time)
{
	return (power * time / 1000.0); // Time is in ms, hence the division by 1000.
}
