#ifndef OPENCL_UTIL_H_
#define OPENCL_UTIL_H_

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#if defined(NV) //NVIDIA
	#include <oclUtils.h>
#elif defined(__APPLE__)
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#ifdef _WIN32
	#define snprintf _snprintf
	#include <malloc.h> //for alligned malloc
#endif

#define STRING_BUFFER_LEN 1024
#define AOCL_ALIGNMENT 64

// For functions that "return" the error code
#define CL_SAFE_CALL(...) do {													\
	cl_int __ret = __VA_ARGS__;												\
	if (__ret != CL_SUCCESS) {												\
		fprintf(stderr, "%s:%d: %s failed with error code ", __FILE__, __LINE__, extractFunctionName(#__VA_ARGS__) );	\
		display_error_message(__ret, stderr);										\
		exit(-1);													\
	}															\
} while (0)

// Declaring some of the functions here to avoid reordering them
inline static char* extractFunctionName(const char* input);
inline static void display_error_message(cl_int errcode, FILE *out);


inline static void device_info_string(cl_device_id device, cl_device_info param, const char* name)
{
	char string[STRING_BUFFER_LEN];
	CL_SAFE_CALL( clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &string, NULL) );
	fprintf(stderr, "%-32s= %s\n", name, string);
}

inline static void device_info_device_type(cl_device_id device, cl_device_info param, const char* name)
{
        cl_device_type device_type;
	CL_SAFE_CALL( clGetDeviceInfo(device, param, sizeof(cl_device_type), &device_type, NULL) );
	fprintf(stderr, "%-32s= %d\n", name, (int)device_type);
}

// Prints memory size in MBytes
inline static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name)
{
        cl_ulong size;
	CL_SAFE_CALL( clGetDeviceInfo(device, param, sizeof(cl_ulong), &size, NULL) );
	if (param == CL_DEVICE_GLOBAL_MEM_SIZE)
	{
		fprintf(stderr, "%-32s= %0.3lf MBytes\n", name, (double)(size/(1024.0*1024.0)));
	}
	else if (param == CL_DEVICE_LOCAL_MEM_SIZE || param == CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)
	{
		fprintf(stderr, "%-32s= %0.3lf KBytes\n", name, (double)(size/1024.0));
	}
}

// Displays available platforms and devices
inline static void display_device_info(cl_platform_id** platforms, cl_uint* platformCount)
{
	unsigned i, j;
	cl_int error;
	cl_uint deviceCount;
	cl_device_id* devices;

	// Get all platforms
	CL_SAFE_CALL( clGetPlatformIDs(0, NULL, platformCount) );
	*platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * (*platformCount));
	CL_SAFE_CALL( clGetPlatformIDs((*platformCount), *platforms, NULL) );
	
	fprintf(stderr, "\nQuerying devices for info:\n");

	for (i = 0; i < *platformCount; i++)
	{
		// Get all devices
		error = clGetDeviceIDs((*platforms)[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
		if ( error != CL_SUCCESS )
		{
			if ( error == CL_DEVICE_NOT_FOUND ) // No compatible OpenCL devices?
			{
				fprintf(stderr, "================================================================================\n");
				fprintf(stderr, "Platform number %d:\n\n", i);
				fprintf(stderr, "No devices were found in this platfrom!\n");
				fprintf(stderr, "================================================================================\n\n");
			}
			else
			{
				fprintf(stderr, "%s:%d: clGetDeviceIDs() failed with error code ", __FILE__, __LINE__);
				display_error_message(error, stderr);
				exit(-1);
			}
		}
		else
		{
			devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
			CL_SAFE_CALL( clGetDeviceIDs((*platforms)[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL) );
			
			for (j = 0; j < deviceCount; j++)
			{
				fprintf(stderr, "================================================================================\n");
				fprintf(stderr, "Platform number %d, device number %d (device count: %d):\n\n", i, j, deviceCount);
				device_info_string(devices[j], CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
				device_info_string(devices[j], CL_DEVICE_NAME, "CL_DEVICE_NAME");
				device_info_string(devices[j], CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
				device_info_ulong(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
				device_info_ulong(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
				device_info_ulong(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
				device_info_device_type(devices[j], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
				fprintf(stderr, "================================================================================\n\n");
			}  
		}
	}
}

// Selects device type based on user input
inline static void select_device_interactive(cl_platform_id* platforms, cl_uint* platformCount, cl_device_type* device_type)
{
	unsigned valid = 0;
	char device_type_char[2];

	fprintf(stderr, "================================================================================\n");
	while ( !valid )
	{
		fprintf(stderr, "Please choose CL_DEVICE_TYPE: ");
		if ( !scanf("%s",device_type_char) )
		{
			fprintf(stderr, "Failed to receive input!\n");
			exit(-1);
		}
		*device_type = (cl_device_type)atoi(device_type_char);
		if ( strlen(device_type_char) > 1 || (!isdigit(device_type_char[0])) )
		{
			fprintf(stderr, "Device type should be a single-digit number!\n");
		}
		else if ( *device_type == CL_DEVICE_TYPE_CPU || *device_type == CL_DEVICE_TYPE_GPU || *device_type == CL_DEVICE_TYPE_ACCELERATOR )
		{
			valid = 1;
		}
		else
		{
			fprintf(stderr, "Invalid device type.\n");
		}
	}
	fprintf(stderr, "================================================================================\n\n");
}

// Checks to see if environmental variable for choosing device type is set or not
inline static void select_device_type(cl_platform_id* platforms, cl_uint* platformCount,
                                      cl_device_type* device_type) {
  char *dev_type_env = getenv("CL_DEVICE_TYPE");
  if (dev_type_env) {
    if (!strcmp(dev_type_env, "CL_DEVICE_TYPE_ALL")) {
      *device_type = CL_DEVICE_TYPE_ALL;
      return;
    } else if (!strcmp(dev_type_env, "CL_DEVICE_TYPE_CPU")) {
      *device_type = CL_DEVICE_TYPE_CPU;
      return;      
    } else if (!strcmp(dev_type_env, "CL_DEVICE_TYPE_GPU")) {
      *device_type = CL_DEVICE_TYPE_GPU;
      return;      
    } else if (!strcmp(dev_type_env, "CL_DEVICE_TYPE_ACCELERATOR")) {
      *device_type = CL_DEVICE_TYPE_ACCELERATOR;
      return;      
    } else {
      fprintf(stderr, "Ignoring invalid device environment variable: %s\n",
              dev_type_env);
    }
  }
  select_device_interactive(platforms, platformCount, device_type);
}

// Validates device type selection and exports context properties
inline static void validate_selection(cl_platform_id* platforms, cl_uint* platformCount, cl_context_properties* ctxprop, cl_device_type* device_type)
{
	unsigned i;
	cl_int error;
	cl_device_id* devices;
	cl_uint deviceCount;
	char deviceName[STRING_BUFFER_LEN];
	
	// Searching for compatible devices based on device_type
	for (i = 0; i < *platformCount; i++)
	{
		error = clGetDeviceIDs(platforms[i], *device_type, 0, NULL, &deviceCount);
		if ( error != CL_SUCCESS )
		{
			if ( error == CL_DEVICE_NOT_FOUND ) // No compatible OpenCL devices?
			{
				fprintf(stderr, "================================================================================\n");
				fprintf(stderr, "Platform number: %d\n", i);
				fprintf(stderr, "No compatible devices found, moving to next platfrom, if any.\n");
				fprintf(stderr, "================================================================================\n\n");
			}
			else
			{
				fprintf(stderr, "%s:%d: clGetDeviceIDs() failed with error code ", __FILE__, __LINE__);
				display_error_message(error, stderr);
				exit(-1);
			}
		}
		else
		{
			devices = (cl_device_id*) malloc(sizeof(cl_device_id) * deviceCount);
			CL_SAFE_CALL( clGetDeviceIDs(platforms[i], *device_type, deviceCount, devices, NULL) );
			CL_SAFE_CALL( clGetDeviceInfo(devices[0], CL_DEVICE_NAME, STRING_BUFFER_LEN, &deviceName, NULL) );
			
			fprintf(stderr, "================================================================================\n");
			fprintf(stderr, "Selected platfrom number: %d\n", i);
			fprintf(stderr, "Device count: %d\n", deviceCount);
			fprintf(stderr, "Device type: %d\n", (int) *device_type);
			fprintf(stderr, "Selected device: %s\n", deviceName);
			fprintf(stderr, "================================================================================\n\n");
			
			ctxprop[0] = CL_CONTEXT_PLATFORM;
			ctxprop[1] = (cl_context_properties)platforms[i];
			ctxprop[2] = 0;
			break;
		}
	}
}

inline static void display_error_message(cl_int errcode, FILE *out) {
  switch (errcode) {
	
	// Common error codes
	case CL_SUCCESS				: fprintf(out, "CL_SUCCESS.\n"); break;
	case CL_DEVICE_NOT_FOUND		: fprintf(out, "CL_DEVICE_NOT_FOUND.\n"); break;
	case CL_DEVICE_NOT_AVAILABLE		: fprintf(out, "CL_DEVICE_NOT_AVAILABLE.\n"); break;
	case CL_COMPILER_NOT_AVAILABLE		: fprintf(out, "CL_COMPILER_NOT_AVAILABLE.\n"); break;
	case CL_MEM_OBJECT_ALLOCATION_FAILURE	: fprintf(out, "CL_MEM_OBJECT_ALLOCATION_FAILURE.\n"); break;
	case CL_OUT_OF_RESOURCES		: fprintf(out, "CL_OUT_OF_RESOURCES.\n"); break;
	case CL_OUT_OF_HOST_MEMORY		: fprintf(out, "CL_OUT_OF_HOST_MEMORY.\n"); break;
	case CL_PROFILING_INFO_NOT_AVAILABLE	: fprintf(out, "CL_PROFILING_INFO_NOT_AVAILABLE.\n"); break;
	case CL_MEM_COPY_OVERLAP		: fprintf(out, "CL_MEM_COPY_OVERLAP.\n"); break;
	case CL_IMAGE_FORMAT_MISMATCH		: fprintf(out, "CL_IMAGE_FORMAT_MISMATCH.\n"); break;
	case CL_IMAGE_FORMAT_NOT_SUPPORTED	: fprintf(out, "CL_IMAGE_FORMAT_NOT_SUPPORTED.\n"); break;
	case CL_BUILD_PROGRAM_FAILURE		: fprintf(out, "CL_BUILD_PROGRAM_FAILURE.\n"); break;
	case CL_MAP_FAILURE			: fprintf(out, "CL_MAP_FAILURE.\n"); break;
	case CL_MISALIGNED_SUB_BUFFER_OFFSET	: fprintf(out, "CL_MISALIGNED_SUB_BUFFER_OFFSET.\n"); break;
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST	: fprintf(out, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST.\n"); break;
#ifdef	AMD
	case CL_COMPILE_PROGRAM_FAILURE		: fprintf(out, "CL_COMPILE_PROGRAM_FAILURE.\n"); break;
	case CL_LINKER_NOT_AVAILABLE		: fprintf(out, "CL_LINKER_NOT_AVAILABLE.\n"); break;
	case CL_LINK_PROGRAM_FAILURE		: fprintf(out, "CL_LINK_PROGRAM_FAILURE.\n"); break;
	case CL_DEVICE_PARTITION_FAILED		: fprintf(out, "CL_DEVICE_PARTITION_FAILED.\n"); break;
	case CL_KERNEL_ARG_INFO_NOT_AVAILABLE	: fprintf(out, "CL_KERNEL_ARG_INFO_NOT_AVAILABLE.\n"); break;
#endif	
	
	case CL_INVALID_VALUE			: fprintf(out, "CL_INVALID_VALUE.\n"); break;
	case CL_INVALID_DEVICE_TYPE		: fprintf(out, "CL_INVALID_DEVICE_TYPE.\n"); break;
	case CL_INVALID_PLATFORM		: fprintf(out, "CL_INVALID_PLATFORM.\n"); break;
	case CL_INVALID_DEVICE			: fprintf(out, "CL_INVALID_DEVICE.\n"); break;
	case CL_INVALID_CONTEXT			: fprintf(out, "CL_INVALID_CONTEXT.\n"); break;
	case CL_INVALID_QUEUE_PROPERTIES	: fprintf(out, "CL_INVALID_QUEUE_PROPERTIES.\n"); break;
	case CL_INVALID_COMMAND_QUEUE		: fprintf(out, "CL_INVALID_COMMAND_QUEUE.\n"); break;
	case CL_INVALID_HOST_PTR		: fprintf(out, "CL_INVALID_HOST_PTR.\n"); break;
	case CL_INVALID_MEM_OBJECT		: fprintf(out, "CL_INVALID_MEM_OBJECT.\n"); break;
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR	: fprintf(out, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR.\n"); break;
	case CL_INVALID_IMAGE_SIZE		: fprintf(out, "CL_INVALID_IMAGE_SIZE.\n"); break;
	case CL_INVALID_SAMPLER			: fprintf(out, "CL_INVALID_SAMPLER.\n"); break;
	case CL_INVALID_BINARY			: fprintf(out, "CL_INVALID_BINARY.\n"); break;
	case CL_INVALID_BUILD_OPTIONS		: fprintf(out, "CL_INVALID_BUILD_OPTIONS.\n"); break;
	case CL_INVALID_PROGRAM			: fprintf(out, "CL_INVALID_PROGRAM.\n"); break;
	case CL_INVALID_PROGRAM_EXECUTABLE	: fprintf(out, "CL_INVALID_PROGRAM_EXECUTABLE.\n"); break;
	case CL_INVALID_KERNEL_NAME		: fprintf(out, "CL_INVALID_KERNEL_NAME.\n"); break;
	case CL_INVALID_KERNEL_DEFINITION	: fprintf(out, "CL_INVALID_KERNEL_DEFINITION.\n"); break;
	case CL_INVALID_KERNEL			: fprintf(out, "CL_INVALID_KERNEL.\n"); break;
	case CL_INVALID_ARG_INDEX		: fprintf(out, "CL_INVALID_ARG_INDEX.\n"); break;
	case CL_INVALID_ARG_VALUE		: fprintf(out, "CL_INVALID_ARG_VALUE.\n"); break;
	case CL_INVALID_ARG_SIZE		: fprintf(out, "CL_INVALID_ARG_SIZE.\n"); break;
	case CL_INVALID_KERNEL_ARGS		: fprintf(out, "CL_INVALID_KERNEL_ARGS.\n"); break;
	case CL_INVALID_WORK_DIMENSION		: fprintf(out, "CL_INVALID_WORK_DIMENSION.\n"); break;
	case CL_INVALID_WORK_GROUP_SIZE		: fprintf(out, "CL_INVALID_WORK_GROUP_SIZE.\n"); break;
	case CL_INVALID_WORK_ITEM_SIZE		: fprintf(out, "CL_INVALID_WORK_ITEM_SIZE.\n"); break;
	case CL_INVALID_GLOBAL_OFFSET		: fprintf(out, "CL_INVALID_GLOBAL_OFFSET.\n"); break;
	case CL_INVALID_EVENT_WAIT_LIST		: fprintf(out, "CL_INVALID_EVENT_WAIT_LIST.\n"); break;
	case CL_INVALID_EVENT			: fprintf(out, "CL_INVALID_EVENT.\n"); break;
	case CL_INVALID_OPERATION		: fprintf(out, "CL_INVALID_OPERATION.\n"); break;
	case CL_INVALID_GL_OBJECT		: fprintf(out, "CL_INVALID_GL_OBJECT.\n"); break;
	case CL_INVALID_BUFFER_SIZE		: fprintf(out, "CL_INVALID_BUFFER_SIZE.\n"); break;
	case CL_INVALID_MIP_LEVEL		: fprintf(out, "CL_INVALID_MIP_LEVEL.\n"); break;
	case CL_INVALID_GLOBAL_WORK_SIZE	: fprintf(out, "CL_INVALID_GLOBAL_WORK_SIZE.\n"); break;
#ifdef AMD	
	case CL_INVALID_PROPERTY		: fprintf(out, "CL_INVALID_PROPERTY.\n"); break;
	case CL_INVALID_IMAGE_DESCRIPTOR	: fprintf(out, "CL_INVALID_IMAGE_DESCRIPTOR.\n"); break;
	case CL_INVALID_COMPILER_OPTIONS	: fprintf(out, "CL_INVALID_COMPILER_OPTIONS.\n"); break;
	case CL_INVALID_LINKER_OPTIONS		: fprintf(out, "CL_INVALID_LINKER_OPTIONS.\n"); break;
	case CL_INVALID_DEVICE_PARTITION_COUNT	: fprintf(out, "CL_INVALID_DEVICE_PARTITION_COUNT.\n"); break;
	case CL_INVALID_PIPE_SIZE		: fprintf(out, "CL_INVALID_PIPE_SIZE.\n"); break;
	case CL_INVALID_DEVICE_QUEUE		: fprintf(out, "CL_INVALID_DEVICE_QUEUE.\n"); break;
#endif	

	/*// DirectX and GL error codes
	case CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR	: fprintf(out, "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR.\n"); break;
	case CL_PLATFORM_NOT_FOUND_KHR			: fprintf(out, "CL_PLATFORM_NOT_FOUND_KHR.\n"); break;
	case CL_INVALID_D3D10_DEVICE_KHR		: fprintf(out, "CL_INVALID_D3D10_DEVICE_KHR.\n"); break;
	case CL_INVALID_D3D10_RESOURCE_KHR		: fprintf(out, "CL_INVALID_D3D10_RESOURCE_KHR.\n"); break;
	case CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR	: fprintf(out, "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR.\n"); break;
	case CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR		: fprintf(out, "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR.\n"); break;
	case CL_INVALID_D3D11_DEVICE_KHR		: fprintf(out, "CL_INVALID_D3D11_DEVICE_KHR.\n"); break;
	case CL_INVALID_D3D11_RESOURCE_KHR		: fprintf(out, "CL_INVALID_D3D11_RESOURCE_KHR.\n"); break;
	case CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR	: fprintf(out, "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR.\n"); break;
	case CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR 	: fprintf(out, "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR .\n"); break;
	case CL_INVALID_DX9_MEDIA_ADAPTER_KHR		: fprintf(out, "CL_INVALID_DX9_MEDIA_ADAPTER_KHR.\n"); break;
	case CL_INVALID_DX9_MEDIA_SURFACE_KHR		: fprintf(out, "CL_INVALID_DX9_MEDIA_SURFACE_KHR.\n"); break;
	case CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR	: fprintf(out, "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR.\n"); break;
	case CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR	: fprintf(out, "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR.\n"); break;*/

	default: fprintf(out, "Unknown OpenCL error code %d!\n", errcode); break;
  }
}

// Extract function name from __VA_ARGS__
inline static char* extractFunctionName(const char* input)
{
	unsigned i;
	char* output = (char*) malloc(strlen(input) * sizeof(char));
	
	for (i = 0; i<strlen(input); i++)
	{
		output[i] = input[i];
		if ( input[i] == '(' )
		{
			break;
		}
	}
	output[i+1] = ')';
	output[i+2] = '\0';
	
	return output;
}

// Safe version of clBuildProgram() that automatically prints compilation log in case of failure
inline static void clBuildProgram_SAFE(cl_program program, cl_uint num_devices, const cl_device_id *device_list, const char *options, void (CL_CALLBACK *pfn_notify) (cl_program program, void *user_data), void *user_data)
{
	cl_int error = clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);

	if (error != CL_SUCCESS)
	{
		fprintf(stderr, "%s:%d: %s failed with error code ", __FILE__, __LINE__, "clBuildProgram()" );
		display_error_message(error, stderr);
		
		if (error == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t log_size;
			char *log;

			// Get log size
			CL_SAFE_CALL( clGetProgramBuildInfo(program, *device_list, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size) );

			// Allocate memory for the log
			log = (char *)malloc(log_size);
			if (log == NULL)
			{
				fprintf(stderr, "Failed to allocate memory for compilation log" );
				exit(-1);
			}

			// Get the log
			CL_SAFE_CALL( clGetProgramBuildInfo(program, *device_list, CL_PROGRAM_BUILD_LOG, log_size, log, NULL) );

			// Print the log
			fprintf(stderr, "\n=============================== Start of compilation log ===============================\n");
			fprintf(stderr, "Build options: %s\n\n", options);
			fprintf(stderr, "%s", log);
			fprintf(stderr, "================================ End of compilation log ================================\n");
		}
		exit(-1);
	}
}

inline static char* getVersionedKernelName(const char* kernel_name,
                                           int version) {
  int slen = strlen(kernel_name) + 32;
  char *vname = (char *)malloc(sizeof(char)*(slen));
  // versioning
#if defined(ALTERA_CL)
  snprintf(vname, slen, "%s_v%d.aocx", kernel_name, version);
#elif defined(USE_JIT)
  snprintf(vname, slen, "%s_v%d.cl", kernel_name, version);
#else
#error "Unsupported."
#endif
  fprintf(stderr, "Using kernel file: %s\n", vname);
  return vname;
}

inline static char* getVersionedKernelName2(const char* kernel_name,
                                            const char* version_string) {
  int slen = strlen(kernel_name) + 128;
  char *vname = (char *)malloc(sizeof(char)*(slen));
  // versioning
#if defined(ALTERA_CL)
  snprintf(vname, slen, "%s_%s.aocx", kernel_name, version_string);
#elif defined(USE_JIT)
  snprintf(vname, slen, "%s_%s.cl", kernel_name, version_string);
#else
#error "Unsupported."
#endif
  fprintf(stderr, "Using kernel file: %s\n", vname);
  return vname;
}


#if 0
inline static void shift_argv(int *argc, char ***argv, int s) {
  int i;
  for (i = 1+s; i < *argc; ++i) {
    argv[i-s] = argv[i];
  }
  --*argc;
}
#endif

inline static void init_fpga(int *argc, char ***argv,
                             int *version) {
  int shift = 0;
  int arg_idx = (*argc) - 1;
  fprintf(stderr, "Initialization\n");
  
  // Default version
  *version = 0;

  if (arg_idx > 0) {
    int ret = sscanf((*argv)[arg_idx], "v%d", version);
    if (ret == 1) {
      ++shift;
    }
  }

  // version number given
  fprintf(stderr, "Using verison %d\n", *version);
  
  //shift_argv(argc, argv, shift);
  *argc -= shift;
  return;
}


inline static void init_fpga2(int *argc, char ***argv,
                              char **version_string,
                              int *version_number) {
  int shift = 0;
  int arg_idx = (*argc) - 1;
  fprintf(stderr, "Initialization\n");

  // Default version
  *version_number = 0;
  // default version string "v0"  
  *version_string = (char*)malloc(sizeof(char)*3);
  (*version_string)[0] = 'v';
  (*version_string)[1] = '0';
  (*version_string)[2] = '\0';

  if (arg_idx > 0) {
    int ret = sscanf((*argv)[arg_idx], "v%d", version_number);
    if (ret == 1) {
      ++shift;
      *version_string = (*argv)[arg_idx];
    }
  }

  // version number given
  fprintf(stderr, "Using verison %d (%s)\n", *version_number,
          *version_string);
  
  //shift_argv(argc, argv, shift);
  *argc -= shift;
  return;
}


inline static int is_ndrange_kernel(int version) {
  return (version % 2) == 0;
}

inline static char* read_kernel(const char *kernel_file_path, size_t *source_size)
{
	// Open kernel file
	FILE *kernel_file;
#ifdef _WIN32
	fopen_s(&kernel_file, kernel_file_path, "rb");
#else
	kernel_file = fopen(kernel_file_path, "rb");
#endif
	if(!kernel_file)
	{
		fprintf(stderr, "Failed to open input kernel file \"%s\".\n", kernel_file_path);
		exit(-1);
	}

	// Detremine the size of the input kernel or binary file
	fseek(kernel_file, 0, SEEK_END);
	*source_size = ftell(kernel_file);
	rewind(kernel_file);
	
	// Allocate memory for the input kernel or binary file
	char *source = (char *)calloc(*source_size + 1, sizeof(char)); 
	if(!source)
	{
		fprintf(stderr, "Failed to allocate enough memory for kernel file.\n");
		exit(-1);
	}
	
	// Read the input kernel or binary file into memory
	if ( !fread(source, 1, *source_size, kernel_file) )
	{
		fprintf(stderr, "Failed to read kernel file into memory.\n");
		exit(-1);
	}
	fclose(kernel_file);
	
	source[*source_size] = '\0';
	return source;
}

inline static void* alignedMalloc(size_t size)
{
#ifdef _WIN32
	void *ptr = _aligned_malloc(size, AOCL_ALIGNMENT);
	if (ptr == NULL)
	{
		fprintf(stderr, "Aligned Malloc failed due to insufficient memory.\n");
		exit(-1);
	}
	return ptr;
#else
	void *ptr = NULL;
	if ( posix_memalign (&ptr, AOCL_ALIGNMENT, size) )
	{
		fprintf(stderr, "Aligned Malloc failed due to insufficient memory.\n");
		exit(-1);
	}
	return ptr;
#endif	
}

inline static void* alignedCalloc(size_t size)
{
#ifdef _WIN32
	void *ptr = _aligned_malloc(size, AOCL_ALIGNMENT);
	if (ptr == NULL)
	{
		fprintf(stderr, "Aligned Calloc failed due to insufficient memory.\n");
		exit(-1);
	}
	memset(ptr, 0, size);
	return ptr;
#else
	void *ptr = NULL;
	if ( posix_memalign (&ptr, AOCL_ALIGNMENT, size) )
	{
		fprintf(stderr, "Aligned Calloc failed due to insufficient memory.\n");
		exit(-1);
	}
	memset(ptr, 0, size);
	return ptr;
#endif
}

#endif /* OPENCL_UTIL_H_ */
