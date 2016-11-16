//------------------------------------------
//--cambine:helper function for OpenCL
//--programmer:	Jianbin Fang
//--date:	27/12/2010
//------------------------------------------
#ifndef _CL_HELPER_
#define _CL_HELPER_

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "../common/opencl_util.h"

using std::string;
using std::ifstream;
using std::cerr;
using std::endl;
using std::cout;
//#pragma OPENCL EXTENSION cl_nv_compiler_options:enable
#define WORK_DIM 2	//work-items dimensions

struct oclHandleStruct
{
    cl_context              context;
    cl_device_id            *devices;
    cl_command_queue        queue;
    cl_program              program;
    cl_int		cl_status;
    std::string error_str;
    std::vector<cl_kernel>  kernel;
};

struct oclHandleStruct oclHandles;

int total_kernels = 2;
string kernel_names[2] = {"BFS_1", "BFS_2"};
size_t work_group_size = 512;
int device_id_inused = 0; //deviced id used (default : 0)

/*
 * Converts the contents of a file into a string
 */
string FileToString(const string fileName)
{
    ifstream f(fileName.c_str(), ifstream::in | ifstream::binary);

    try
    {
        size_t size;
        char*  str;
        string s;

        if(f.is_open())
        {
            size_t fileSize;
            f.seekg(0, ifstream::end);
            size = fileSize = f.tellg();
            f.seekg(0, ifstream::beg);

            str = new char[size+1];
            if (!str) throw(string("Could not allocate memory"));

            f.read(str, fileSize);
            f.close();
            str[size] = '\0';
        
            s = str;
            delete [] str;
            return s;
        }
    }
    catch(std::string msg)
    {
        cerr << "Exception caught in FileToString(): " << msg << endl;
        if(f.is_open())
            f.close();
    }
    catch(...)
    {
        cerr << "Exception caught in FileToString()" << endl;
        if(f.is_open())
            f.close();
    }
    string errorMsg = "FileToString()::Error: Unable to open file "
                            + fileName;
    throw(errorMsg);
}
//---------------------------------------
//Read command line parameters
//
void _clCmdParams(int argc, char* argv[]){
	for (int i =0; i < argc; ++i)
	{
		switch (argv[i][1])
		{
		case 'g':	//--g stands for size of work group
			if (++i < argc)
			{
			// Altera's ARM compiler considers size_t as unsigned while other compilers say it is unsigned long
			#ifdef ARM
				sscanf(argv[i], "%u", &work_group_size);
			#else
				sscanf(argv[i], "%lu", &work_group_size);
			#endif
			}
			else
			{
				std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
				throw;
			}
			break;
		  case 'd':	 //--d stands for device id used in computaion
			if (++i < argc)
			{
				sscanf(argv[i], "%u", &device_id_inused);
			}
			else
			{
				std::cerr << "Could not read argument after option " << argv[i-1] << std::endl;
				throw;
			}
			break;
		default:
			;
		}
	}
	
}

//---------------------------------------
//Initlize CL objects
//--description: there are 5 steps to initialize all the OpenCL objects needed
//--revised on 04/01/2011: get the number of devices  and 
//  devices have no relationship with context
void _clInit(int version)
{
    int DEVICE_ID_INUSED = device_id_inused;
    size_t size;
    cl_int resultCL;
    cl_context_properties cprops[3];
    cl_uint numPlatforms;
    cl_platform_id* targetPlatform = NULL;
    cl_device_type device_type;
    cl_uint deviceListSize;
    
    oclHandles.context = NULL;
    oclHandles.devices = NULL;
    oclHandles.queue = NULL;
    oclHandles.program = NULL;

    //-----------------------------------------------
    //--cambine-1: find the available platforms and select one

    display_device_info(&targetPlatform, &numPlatforms);
    select_device_type(targetPlatform, &numPlatforms, &device_type);
    validate_selection(targetPlatform, &numPlatforms, cprops, &device_type);

    //-----------------------------------------------
    //--cambine-2: create an OpenCL context

    oclHandles.context = clCreateContextFromType(cprops, 
                                                device_type, 
                                                NULL, 
                                                NULL, 
                                                &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.context == NULL))
        throw (string("InitCL()::Error: Creating Context (clCreateContextFromType)"));
   //-----------------------------------------------
   //--cambine-3: detect OpenCL devices	
    
   // First, get the size of device list 
   oclHandles.cl_status = clGetContextInfo( oclHandles.context, CL_CONTEXT_DEVICES, 0, NULL, &size );
   deviceListSize = (int) (size / sizeof(cl_device_id));
   if(oclHandles.cl_status!=CL_SUCCESS)
        throw(string("exception in _clInit -> clGetContextInfo"));

    //std::cout<<"device number:"<<deviceListSize<<std::endl;

    // Now, allocate the device list 
   oclHandles.devices = (cl_device_id *)malloc(deviceListSize * sizeof(cl_device_id));
   if (oclHandles.devices == 0)
        throw(string("InitCL()::Error: Could not allocate memory."));

   // Next, get the device list data 
   oclHandles.cl_status = clGetContextInfo( oclHandles.context, CL_CONTEXT_DEVICES, size, oclHandles.devices, NULL );
   if(oclHandles.cl_status!=CL_SUCCESS){
   	throw(string("exception in _clInit -> clGetContextInfo-2"));   	
   }
    
   //-----------------------------------------------
   //--cambine-4: Create an OpenCL command queue    
    oclHandles.queue = clCreateCommandQueue(oclHandles.context, 
                                            oclHandles.devices[DEVICE_ID_INUSED], 
                                            CL_QUEUE_PROFILING_ENABLE, 
                                            &resultCL);

    if ((resultCL != CL_SUCCESS) || (oclHandles.queue == NULL))
        throw(string("InitCL()::Creating Command Queue. (clCreateCommandQueue)"));
    //-----------------------------------------------
    //--cambine-5: Load CL file, build CL program object, create CL kernel object
    
    size_t sourceSize;
    char *kernel_file_path = getVersionedKernelName("./bfs_kernel", version);
    char *source = read_kernel(kernel_file_path, &sourceSize);
    
    // Create the program object
#ifdef USE_JIT
    oclHandles.program = clCreateProgramWithSource(oclHandles.context, 1, (const char **)&source, NULL, &resultCL);
#else
    oclHandles.program = clCreateProgramWithBinary(oclHandles.context, 1, oclHandles.devices, &sourceSize, (const unsigned char**)&source, NULL, &resultCL);  
#endif

    if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
        throw(string("InitCL()::Error: Loading Binary into cl_program. (clCreateProgramWithBinary)"));    
    //insert debug information
    //std::string options= "-cl-nv-verbose"; //Doesn't work on AMD machines
    //options += " -cl-nv-opt-level=3";
    const char* compileOptions = "-I .";
    clBuildProgram_SAFE(oclHandles.program, deviceListSize, oclHandles.devices, compileOptions, NULL,NULL);

    /*if ((resultCL != CL_SUCCESS) || (oclHandles.program == NULL))
    {
        cerr << "InitCL()::Error: In clBuildProgram" << endl;

		size_t length;
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[DEVICE_ID_INUSED], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        0, 
                                        NULL, 
                                        &length);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		char* buffer = (char*)malloc(length);
        resultCL = clGetProgramBuildInfo(oclHandles.program, 
                                        oclHandles.devices[DEVICE_ID_INUSED], 
                                        CL_PROGRAM_BUILD_LOG, 
                                        length, 
                                        buffer, 
                                        NULL);
        if(resultCL != CL_SUCCESS) 
            throw(string("InitCL()::Error: Getting Program build info(clGetProgramBuildInfo)"));

		cerr << buffer << endl;
        free(buffer);

        throw(string("InitCL()::Error: Building Program (clBuildProgram)"));
    }*/

    //get program information in intermediate representation
    #ifdef PTX_MSG    
    size_t binary_sizes[deviceListSize];
    char * binaries[deviceListSize];
    unsigned i;
    //figure out number of devices and the sizes of the binary for each device. 
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*deviceListSize, &binary_sizes, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS){
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-2"));
    }

    std::cout<<"--cambine:"<<binary_sizes<<std::endl;
    //copy over all of the generated binaries. 
    for(i=0;i<deviceListSize;i++)
	binaries[i] = (char *)malloc( sizeof(char)*(binary_sizes[i]+1));
    oclHandles.cl_status = clGetProgramInfo(oclHandles.program, CL_PROGRAM_BINARIES, sizeof(char *)*deviceListSize, binaries, NULL );
    if(oclHandles.cl_status!=CL_SUCCESS){
        throw(string("--cambine:exception in _InitCL -> clGetProgramInfo-3"));
    }
    for(i=0;i<deviceListSize;i++)
      binaries[i][binary_sizes[i]] = '\0';
    std::cout<<"--cambine:writing ptd information..."<<std::endl;
    FILE * ptx_file = fopen("cl.ptx","w");
    if(ptx_file==NULL){
	throw(string("exceptions in allocate ptx file."));
    }
    fprintf(ptx_file,"%s",binaries[DEVICE_ID_INUSED]);
    fclose(ptx_file);
    std::cout<<"--cambine:writing ptd information done."<<std::endl;
    for(i=0;i<deviceListSize;i++)
	free(binaries[i]);
    #endif

    for (int nKernel = 0; nKernel < total_kernels; nKernel++)
    {
        /* get a kernel object handle for a kernel with the given name */
        cl_kernel kernel = clCreateKernel(oclHandles.program,
                                            (kernel_names[nKernel]).c_str(),
                                            &resultCL);

        if ((resultCL != CL_SUCCESS) || (kernel == NULL))
        {
            string errorMsg = "InitCL()::Error: Creating Kernel (clCreateKernel) \"" + kernel_names[nKernel] + "\"";
            throw(errorMsg);
        }

        oclHandles.kernel.push_back(kernel);
    }
  //get resource alocation information
    #ifdef RES_MSG
    char * build_log;
    size_t ret_val_size;
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
    if(oclHandles.cl_status!=CL_SUCCESS){
	throw(string("exceptions in _InitCL -> getting resource information"));
    }    

    build_log = (char *)malloc(ret_val_size+1);
    oclHandles.cl_status = clGetProgramBuildInfo(oclHandles.program, oclHandles.devices[DEVICE_ID_INUSED], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
    if(oclHandles.cl_status!=CL_SUCCESS){
	throw(string("exceptions in _InitCL -> getting resources allocation information-2"));
    }
    build_log[ret_val_size] = '\0';
    std::cout<<"--cambine:"<<build_log<<std::endl;
    free(build_log);
    free(source);
    #endif
}

//---------------------------------------
//release CL objects
void _clRelease()
{
    char errorFlag = false;

    for (unsigned nKernel = 0; nKernel < oclHandles.kernel.size(); nKernel++)
    {
        if (oclHandles.kernel[nKernel] != NULL)
        {
            cl_int resultCL = clReleaseKernel(oclHandles.kernel[nKernel]);
            if (resultCL != CL_SUCCESS)
            {
                cerr << "ReleaseCL()::Error: In clReleaseKernel" << endl;
                errorFlag = true;
            }
            oclHandles.kernel[nKernel] = NULL;
        }
        oclHandles.kernel.clear();
    }

    if (oclHandles.program != NULL)
    {
        cl_int resultCL = clReleaseProgram(oclHandles.program);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseProgram" << endl;
            errorFlag = true;
        }
        oclHandles.program = NULL;
    }

    if (oclHandles.queue != NULL)
    {
        cl_int resultCL = clReleaseCommandQueue(oclHandles.queue);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseCommandQueue" << endl;
            errorFlag = true;
        }
        oclHandles.queue = NULL;
    }

    free(oclHandles.devices);

    if (oclHandles.context != NULL)
    {
        cl_int resultCL = clReleaseContext(oclHandles.context);
        if (resultCL != CL_SUCCESS)
        {
            cerr << "ReleaseCL()::Error: In clReleaseContext" << endl;
            errorFlag = true;
        }
        oclHandles.context = NULL;
    }

    if (errorFlag) throw(string("ReleaseCL()::Error encountered."));
}
//--------------------------------------------------------
//--cambine:create buffer and then copy data from host to device
cl_mem _clCreateAndCpyMem(int size, void * h_mem_source) throw(string){
	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,  \
									size, h_mem_source, &oclHandles.cl_status);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem()"));
	return d_mem;
}
//-------------------------------------------------------
//--cambine:	create read only  buffer for devices
//--date:	17/01/2011	
cl_mem _clMallocRW(int size, void * h_mem_ptr) throw(string){
 	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clMallocRW"));
	return d_mem;
}
//-------------------------------------------------------
//--cambine:	create read and write buffer for devices
//--date:	17/01/2011	
cl_mem _clMalloc(int size, void * h_mem_ptr) throw(string){
 	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, size, h_mem_ptr, &oclHandles.cl_status);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clMalloc"));
	return d_mem;
}

//-------------------------------------------------------
//--cambine:	transfer data from host to device
//--date:	17/01/2011
void _clMemcpyH2D(cl_mem d_mem, int size, const void *h_mem_ptr) throw(string){
	oclHandles.cl_status = clEnqueueWriteBuffer(oclHandles.queue, d_mem, CL_TRUE, 0, size, h_mem_ptr, 0, NULL, NULL);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clMemcpyH2D"));
}
//--------------------------------------------------------
//--cambine:create buffer and then copy data from host to device with pinned 
// memory
cl_mem _clCreateAndCpyPinnedMem(int size, float* h_mem_source) throw(string){
	cl_mem d_mem, d_mem_pinned;
	float * h_mem_pinned = NULL;
	d_mem_pinned = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,  \
									size, NULL, &oclHandles.cl_status);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem()->d_mem_pinned"));
	//------------
	d_mem = clCreateBuffer(oclHandles.context,	CL_MEM_READ_ONLY,  \
									size, NULL, &oclHandles.cl_status);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem() -> d_mem "));
	//----------
	h_mem_pinned = (cl_float *)clEnqueueMapBuffer(oclHandles.queue, d_mem_pinned, CL_TRUE,  \
										CL_MAP_WRITE, 0, size, 0, NULL,  \
										NULL,  &oclHandles.cl_status);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem() -> clEnqueueMapBuffer"));
	int element_number = size/sizeof(float);
	#pragma omp parallel for
	for(int i=0;i<element_number;i++){
		h_mem_pinned[i] = h_mem_source[i];
	}
	//----------
	oclHandles.cl_status = clEnqueueWriteBuffer(oclHandles.queue, d_mem, 	\
									CL_TRUE, 0, size, h_mem_pinned,  \
									0, NULL, NULL);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateAndCpyMem() -> clEnqueueWriteBuffer"));
	
	return d_mem;
}


//--------------------------------------------------------
//--cambine:create write only buffer on device
cl_mem _clMallocWO(int size) throw(string){
	cl_mem d_mem;
	d_mem = clCreateBuffer(oclHandles.context, CL_MEM_WRITE_ONLY, size, 0, &oclHandles.cl_status);
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(string("excpetion in _clCreateMem()"));
	return d_mem;
}

//--------------------------------------------------------
//transfer data from device to host
void _clMemcpyD2H(cl_mem d_mem, int size, void * h_mem) throw(string){
	oclHandles.cl_status = clEnqueueReadBuffer(oclHandles.queue, d_mem, CL_TRUE, 0, size, h_mem, 0,0,0);
	oclHandles.error_str = "excpetion in _clCpyMemD2H -> ";
	switch(oclHandles.cl_status){
		case CL_INVALID_COMMAND_QUEUE:
			oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
			break;
		case CL_INVALID_CONTEXT:
			oclHandles.error_str += "CL_INVALID_CONTEXT";
			break;	
		case CL_INVALID_MEM_OBJECT:
			oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
			break;
		case CL_INVALID_VALUE:
			oclHandles.error_str += "CL_INVALID_VALUE";
			break;	
		case CL_INVALID_EVENT_WAIT_LIST:
			oclHandles.error_str += "CL_INVALID_EVENT_WAIT_LIST";
			break;
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			oclHandles.error_str += "CL_MEM_OBJECT_ALLOCATION_FAILURE";
			break;	
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;		
		default:
			oclHandles.error_str += "Unknown reason";
			break;
	}
	if(oclHandles.cl_status != CL_SUCCESS)
		throw(oclHandles.error_str);
}

//--------------------------------------------------------
//set kernel arguments
void _clSetArgs(int kernel_id, int arg_idx, void * d_mem, int size = 0) throw(string){
	if(!size){
		oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, sizeof(d_mem), &d_mem);
		oclHandles.error_str = "excpetion in _clSetKernelArg() ";
		switch(oclHandles.cl_status){
			case CL_INVALID_KERNEL:
				oclHandles.error_str += "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_ARG_INDEX:
				oclHandles.error_str += "CL_INVALID_ARG_INDEX";
				break;	
			case CL_INVALID_ARG_VALUE:
				oclHandles.error_str += "CL_INVALID_ARG_VALUE";
				break;
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;	
			case CL_INVALID_SAMPLER:
				oclHandles.error_str += "CL_INVALID_SAMPLER";
				break;
			case CL_INVALID_ARG_SIZE:
				oclHandles.error_str += "CL_INVALID_ARG_SIZE";
				break;	
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}
		if(oclHandles.cl_status != CL_SUCCESS)
			throw(oclHandles.error_str);
	}
	else{
		oclHandles.cl_status = clSetKernelArg(oclHandles.kernel[kernel_id], arg_idx, size, d_mem);
		oclHandles.error_str = "excpetion in _clSetKernelArg() ";
		switch(oclHandles.cl_status){
			case CL_INVALID_KERNEL:
				oclHandles.error_str += "CL_INVALID_KERNEL";
				break;
			case CL_INVALID_ARG_INDEX:
				oclHandles.error_str += "CL_INVALID_ARG_INDEX";
				break;	
			case CL_INVALID_ARG_VALUE:
				oclHandles.error_str += "CL_INVALID_ARG_VALUE";
				break;
			case CL_INVALID_MEM_OBJECT:
				oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
				break;	
			case CL_INVALID_SAMPLER:
				oclHandles.error_str += "CL_INVALID_SAMPLER";
				break;
			case CL_INVALID_ARG_SIZE:
				oclHandles.error_str += "CL_INVALID_ARG_SIZE";
				break;	
			case CL_OUT_OF_RESOURCES:
				oclHandles.error_str += "CL_OUT_OF_RESOURCES";
				break;
			case CL_OUT_OF_HOST_MEMORY:
				oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
				break;
			default:
				oclHandles.error_str += "Unknown reason";
				break;
		}
		if(oclHandles.cl_status != CL_SUCCESS)
			throw(oclHandles.error_str);
	}
}
void _clFinish() throw(string){
	oclHandles.cl_status = clFinish(oclHandles.queue);	
	oclHandles.error_str = "excpetion in _clFinish";
	switch(oclHandles.cl_status){
		case CL_INVALID_COMMAND_QUEUE:
			oclHandles.error_str += "CL_INVALID_COMMAND_QUEUE";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;		
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;
		default:
			oclHandles.error_str += "Unknown reasons";
			break;

	}
	if(oclHandles.cl_status!=CL_SUCCESS){
		throw(oclHandles.error_str);
	}
}
//--------------------------------------------------------
//--cambine:enqueue kernel
void _clInvokeKernel(int kernel_id, size_t work_items, size_t work_group_size, cl_event* kernel_event, int version) throw(string){
	cl_uint work_dim = WORK_DIM;
	if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	work_items = work_items + (work_group_size-(work_items%work_group_size));
	size_t local_work_size[] = {work_group_size, 1};
	size_t global_work_size[] = {work_items, 1};
	
	if (is_ndrange_kernel(version))
	{
		CL_SAFE_CALL( clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, global_work_size, local_work_size, 0 , 0, kernel_event) );
	}
	else
	{
		CL_SAFE_CALL( clEnqueueTask(oclHandles.queue, oclHandles.kernel[kernel_id], 0, NULL, kernel_event) );
	}

	//_clFinish();
	// oclHandles.cl_status = clWaitForEvents(1, &e[0]);
	// #ifdef ERRMSG
        // if (oclHandles.cl_status!= CL_SUCCESS)
        //     throw(string("excpetion in _clEnqueueNDRange() -> clWaitForEvents"));
	// #endif
}
/*void _clInvokeKernel2D(int kernel_id, size_t range_x, size_t range_y, size_t group_x, size_t group_y) throw(string){
	cl_uint work_dim = WORK_DIM;
	size_t local_work_size[] = {group_x, group_y};
	size_t global_work_size[] = {range_x, range_y};
	cl_event e[1];
	//if(work_items%work_group_size != 0)	//process situations that work_items cannot be divided by work_group_size
	//work_items = work_items + (work_group_size-(work_items%work_group_size));

	if (is_ndrange_kernel(version))
	{
		#ifdef ERRMSG
			CL_SAFE_CALL( clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, global_work_size, local_work_size, 0 , 0, &(e[0])) );
		#else
			clEnqueueNDRangeKernel(oclHandles.queue, oclHandles.kernel[kernel_id], work_dim, 0, global_work_size, local_work_size, 0 , 0, &(e[0]));
		#endif
	}
	else
	{
		#ifdef ERRMSG
			CL_SAFE_CALL( clEnqueueTask(oclHandles.queue, oclHandles.kernel[kernel_id], 0, NULL, &(e[0])) );
		#else
			clEnqueueTask(oclHandles.queue, oclHandles.kernel[kernel_id], 0, NULL, &(e[0]));
		#endif		
	}

	//_clFinish();
	oclHandles.cl_status = clWaitForEvents(1, &e[0]);

	#ifdef ERRMSG

        if (oclHandles.cl_status!= CL_SUCCESS)

            throw(string("excpetion in _clEnqueueNDRange() -> clWaitForEvents"));

	#endif
}*/

//--------------------------------------------------------
//release OpenCL objects
void _clFree(cl_mem ob) throw(string){
	if(ob!=NULL)
		oclHandles.cl_status = clReleaseMemObject(ob);	
	oclHandles.error_str = "excpetion in _clFree() ->";
	switch(oclHandles.cl_status)
	{
		case CL_INVALID_MEM_OBJECT:
			oclHandles.error_str += "CL_INVALID_MEM_OBJECT";
			break;
		case CL_OUT_OF_RESOURCES:
			oclHandles.error_str += "CL_OUT_OF_RESOURCES";
			break;
		case CL_OUT_OF_HOST_MEMORY:
			oclHandles.error_str += "CL_OUT_OF_HOST_MEMORY";
			break;			
		default: 
			oclHandles.error_str += "Unkown reseason";
			break;		
	}        
    if (oclHandles.cl_status!= CL_SUCCESS)
       throw(oclHandles.error_str);
}
#endif //_CL_HELPER_
