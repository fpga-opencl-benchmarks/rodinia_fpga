#include <cstdlib>
#include <cassert>
#include "OpenCL.h"

#include "../common/opencl_util.h"

cl_uint devices_n = 1;

OpenCL::OpenCL(int displayOutput)
{
	VERBOSE = displayOutput;
}

OpenCL::~OpenCL()
{
	// Flush and kill the command queue...
	clFlush(command_queue);
	clFinish(command_queue);
	
	// Release each kernel in the map kernelArray
	map<string, cl_kernel>::iterator it;
	for ( it=kernelArray.begin() ; it != kernelArray.end(); it++ )
		clReleaseKernel( (*it).second );
		
	// Now the program...
	clReleaseProgram(program);
	
	// ...and finally, the queue and context.
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
}

/*size_t OpenCL::localSize()
{
	return this->lwsize;
}*/

cl_command_queue OpenCL::q()
{
	return this->command_queue;
}

void OpenCL::launch(string toLaunch, int version)
{
	if (is_ndrange_kernel(version))
	{
		CL_SAFE_CALL( clEnqueueNDRangeKernel(command_queue, kernelArray[toLaunch], 1, NULL, &gwsize, &lwsize, 0, NULL, NULL) );
	}
	else
	{
		CL_SAFE_CALL( clEnqueueTask(command_queue, kernelArray[toLaunch], 0, NULL, NULL) );
	}
}

void OpenCL::gwSize(size_t theSize)
{
	this->gwsize = theSize;
}

void OpenCL::lwSize(size_t theSize)
{
	this->lwsize = theSize;
}

cl_context OpenCL::ctxt()
{
	return this->context;
}

cl_kernel OpenCL::kernel(string kernelName)
{
	return this->kernelArray[kernelName];
}

void OpenCL::createKernel(string kernelName)
{
	cl_kernel kernel = clCreateKernel(this->program, kernelName.c_str(), NULL);
	kernelArray[kernelName] = kernel;
	
	// Get the kernel work group size.
/*#if !defined(ALTERA_CL)
	CL_SAFE_CALL(clGetKernelWorkGroupInfo(kernelArray[kernelName], device_id[0], CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &lwsize, NULL));
	if (lwsize == 0)
	{
		cout << "Error: clGetKernelWorkGroupInfo() returned a max work group size of zero!" << endl;
		exit(1);
	}
	// Local work size must divide evenly into global work size.
	size_t howManyThreads = lwsize;
	if (lwsize > gwsize)
	{
		lwsize = gwsize;
		printf("Using %zu for local work size. \n", lwsize);
	}
	else
	{
		while (gwsize % howManyThreads != 0)
		{
			howManyThreads--;
		}
		if (VERBOSE)
			printf("Max local threads is %zu. Using %zu for local work size. \n", lwsize, howManyThreads);

		this->lwsize = howManyThreads;
	}
#else        
        // The above clGetKernelWorkGroupInfo call retuns 2^32-1 with
        // AOCL (emulation mode). Not sure it actually
        // works as intended. Workaround by setting the value
        // manually. LWSIZE is given as a CPP macro. We assume this
        // value in the kernel code since the local memory size is
        // also statically given. 
        lwsize = LWSIZE;
        assert(gwsize % lwsize == 0);
        assert(gwsize >= lwsize);
#endif*/
}

void OpenCL::buildKernel(int version)
{
	size_t size;
	char compileOptions[1024];
	
	char *kernel_file_path = getVersionedKernelName("./pathfinder_kernel", version);
	char *source = read_kernel(kernel_file_path, &size);

	// Create a program from the kernel source.
#ifdef USE_JIT
	program = clCreateProgramWithSource(context, 1, (const char **) &source, NULL, &ret);
#else
	program = clCreateProgramWithBinary(context, 1, devices, &size, (const unsigned char **) &source, NULL, &ret);
        
#endif
        
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateProgramWithSource/Binary! Error code %i\n\n", ret);
                display_error_message(ret, stderr);
		exit(1);
	}

	// Memory cleanup for the variable used to hold the kernel source.
	free(source);
	
	// Build (compile) the program.
	sprintf(compileOptions,"-I.");
	clBuildProgram_SAFE(program, devices_n, devices, compileOptions, NULL, NULL);
	
	/*if (ret != CL_SUCCESS)
	{
		printf("\nError at clBuildProgram! Error code %i\n\n", ret);
		cout << "\n*************************************************" << endl;
		cout << "***   OUTPUT FROM COMPILING THE KERNEL FILE   ***" << endl;
		cout << "*************************************************" << endl;
		// Shows the log
		char*  build_log;
		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << endl;
		delete[] build_log;
		cout << "\n*************************************************" << endl;
		cout << "*** END OUTPUT FROM COMPILING THE KERNEL FILE ***" << endl;
		cout << "*************************************************\n\n" << endl;
		exit(1);
	}


	// Show error info from building the program.
	if (VERBOSE)
	{
		cout << "\n*************************************************" << endl;
		cout << "***   OUTPUT FROM COMPILING THE KERNEL FILE   ***" << endl;
		cout << "*************************************************" << endl;
		// Shows the log
		char*  build_log;
		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log = new char[log_size + 1];
		// Second call to get the log
		clGetProgramBuildInfo(program, device_id[0], CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
		build_log[log_size] = '\0';
		cout << build_log << endl;
		delete[] build_log;
		cout << "\n*************************************************" << endl;
		cout << "*** END OUTPUT FROM COMPILING THE KERNEL FILE ***" << endl;
		cout << "*************************************************\n\n" << endl;
	}*/
}

void OpenCL::getDevices()
{
	cl_uint         platforms_n = 0;
	cl_device_type  deviceType;
	cl_platform_id *platforms = NULL;
	cl_context_properties ctxprop[3];
	size_t size;
#if 0	
	/* The following code queries the number of platforms and devices, and
	 * lists the information about both.
	 */
	clGetPlatformIDs(100, platform_id, &platforms_n);
	if (VERBOSE)
	{
		printf("\n=== %d OpenCL platform(s) found: ===\n", platforms_n);
		for (int i = 0; i < platforms_n; i++)
		{
			char buffer[10240];
			printf("  -- %d --\n", i);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_PROFILE, 10240, buffer,
			                  NULL);
			printf("  PROFILE = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VERSION, 10240, buffer,
			                  NULL);
			printf("  VERSION = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_NAME, 10240, buffer, NULL);
			printf("  NAME = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_VENDOR, 10240, buffer, NULL);
			printf("  VENDOR = %s\n", buffer);
			clGetPlatformInfo(platform_id[i], CL_PLATFORM_EXTENSIONS, 10240, buffer,
			                  NULL);
			printf("  EXTENSIONS = %s\n", buffer);
		}
	}
	
	clGetDeviceIDs(platform_id[0], deviceType, 100, device_id, &devices_n);
	if (VERBOSE)
	{
		printf("Using the default platform (platform 0)...\n\n");
		printf("=== %d OpenCL device(s) found on platform:\n", devices_n);
		for (int i = 0; i < devices_n; i++)
		{
			char buffer[10240];
			cl_uint buf_uint;
			cl_ulong buf_ulong;
			printf("  -- %d --\n", i);
			clGetDeviceInfo(device_id[i], CL_DEVICE_NAME, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_NAME = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VENDOR, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_VENDOR = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_VERSION, sizeof(buffer), buffer,
			                NULL);
			printf("  DEVICE_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DRIVER_VERSION, sizeof(buffer), buffer,
			                NULL);
			printf("  DRIVER_VERSION = %s\n", buffer);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_COMPUTE_UNITS,
			                sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_MAX_CLOCK_FREQUENCY,
			                sizeof(buf_uint), &buf_uint, NULL);
			printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int) buf_uint);
			clGetDeviceInfo(device_id[i], CL_DEVICE_GLOBAL_MEM_SIZE,
			                sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n",
			       (unsigned long long) buf_ulong);
			clGetDeviceInfo(device_id[i], CL_DEVICE_LOCAL_MEM_SIZE,
			                sizeof(buf_ulong), &buf_ulong, NULL);
			printf("  CL_DEVICE_LOCAL_MEM_SIZE = %llu\n",
			       (unsigned long long) buf_ulong);
		}
		printf("\n");
	}
#else
	display_device_info(&platforms, &platforms_n);
	select_device_type(platforms, &platforms_n, &deviceType);
	validate_selection(platforms, &platforms_n, ctxprop, &deviceType);
#endif
	
	// Create an OpenCL context.
	context = clCreateContextFromType(ctxprop, deviceType, NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateContext! Error code %i\n\n", ret);
		exit(1);
	}
	
	CL_SAFE_CALL( clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size) );
	devices_n = (int) (size / sizeof(cl_device_id));
	devices = (cl_device_id *)malloc(size);
	if (devices == NULL)
	{
		printf("\nFailed to allocate memory for devices.\n\n");
		exit(1);
	}
	CL_SAFE_CALL( clGetContextInfo( context, CL_CONTEXT_DEVICES, size, devices, NULL ) );
 
	// Create a command queue.
	command_queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &ret);
	if (ret != CL_SUCCESS)
	{
		printf("\nError at clCreateCommandQueue! Error code %i\n\n", ret);
		exit(1);
	}
}

void OpenCL::init(int version)
{
	getDevices();
	buildKernel(version);
}
