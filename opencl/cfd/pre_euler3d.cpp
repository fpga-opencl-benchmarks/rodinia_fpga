/********************************************************************
	euler3d.cpp
	: parallelized code of CFD
	
	- original code from the AIAA-2009-4001 by Andrew Corrigan, acorriga@gmu.edu
	- parallelization with OpenCL API has been applied by
	Jianbin Fang - j.fang@tudelft.nl
	Delft University of Technology
	Faculty of Electrical Engineering, Mathematics and Computer Science
	Department of Software Technology
	Parallel and Distributed Systems Group
	on 24/03/2011
********************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
//#include "CLHelper.h"
#include "CLHelper2.h" 
#include "common.h"
#include "host_common.h"
#include "../../common/timer.h"

#define KERNEL_PREFIX "./pre_euler3d_kernel"

static bool compute_flux_contributions_arg_set = false;

void compute_flux_contributions(int nelr,
                                cl_mem variables,
                                cl_mem flux_contribution_momentum_x,
                                cl_mem flux_contribution_momentum_y,
                                cl_mem flux_contribution_momentum_z,
                                cl_mem flux_contribution_density_energy) {
  int work_items = nelr;
  int work_group_size = block_size;
  cl_kernel kernel = kernels[compute_flux_contributions_idx];
  if (!compute_flux_contributions_arg_set) {
    int arg_idx = 0;
    _clSetArgs(kernel, arg_idx++, &nelr, sizeof(int));  
    _clSetArgs(kernel, arg_idx++, variables);
    _clSetArgs(kernel, arg_idx++, flux_contribution_momentum_x);
    _clSetArgs(kernel, arg_idx++, flux_contribution_momentum_y);  
    _clSetArgs(kernel, arg_idx++, flux_contribution_momentum_z);  
    _clSetArgs(kernel, arg_idx++, flux_contribution_density_energy);
    compute_flux_contributions_arg_set = true;
  }
  if (is_ndrange_kernel(version_number)) {
    _clInvokeKernel(kernel, work_items, work_group_size);
  } else {
    _clInvokeKernel(kernel);
  }
}

static bool compute_flux_arg_set = false;

void compute_flux(int nelr, cl_mem elements_surrounding_elements, cl_mem normals, cl_mem variables, cl_mem ff_variable, \
                  cl_mem fluxes, cl_mem ff_flux_contribution_density_energy,
                  cl_mem ff_flux_contribution_momentum_x,
                  cl_mem ff_flux_contribution_momentum_y,
                  cl_mem ff_flux_contribution_momentum_z,
                  // pre-computed arrays
                  cl_mem flux_contribution_momentum_x,
                  cl_mem flux_contribution_momentum_y,
                  cl_mem flux_contribution_momentum_z,
                  cl_mem flux_contribution_density_energy) {
  int work_items = nelr;
  int work_group_size = block_size;
  cl_kernel kernel = kernels[compute_flux_idx];
  if (!compute_flux_arg_set) {
    int arg_idx = 0;
    _clSetArgs(kernel, arg_idx++, elements_surrounding_elements);
    _clSetArgs(kernel, arg_idx++, normals);
    _clSetArgs(kernel, arg_idx++, variables);
    _clSetArgs(kernel, arg_idx++, ff_variable);
    _clSetArgs(kernel, arg_idx++, fluxes);
    _clSetArgs(kernel, arg_idx++, ff_flux_contribution_density_energy);
    _clSetArgs(kernel, arg_idx++, ff_flux_contribution_momentum_x);
    _clSetArgs(kernel, arg_idx++, ff_flux_contribution_momentum_y);
    _clSetArgs(kernel, arg_idx++, ff_flux_contribution_momentum_z);
    _clSetArgs(kernel, arg_idx++, flux_contribution_momentum_x);
    _clSetArgs(kernel, arg_idx++, flux_contribution_momentum_y);  
    _clSetArgs(kernel, arg_idx++, flux_contribution_momentum_z);  
    _clSetArgs(kernel, arg_idx++, flux_contribution_density_energy);
    _clSetArgs(kernel, arg_idx++, &nelr, sizeof(int));
    compute_flux_arg_set = true;
  }
  if (is_ndrange_kernel(version_number)) {  
    _clInvokeKernel(kernel, work_items, work_group_size);
  } else {
    _clInvokeKernel(kernel);
  }
}


static void setup() {
  kernel_names.push_back("compute_step_factor");  
  compute_step_factor_idx = 0;
  kernel_names.push_back("time_step");  
  time_step_idx = 1;
  kernel_names.push_back("compute_flux_contributions");  
  compute_flux_contributions_idx = 2;
  kernel_names.push_back("compute_flux");  
  compute_flux_idx = 3;
}
  

/*
 * Main function
 */
int main(int argc, char** argv){
  setup();
  char *vs;
  init_fpga2(&argc, &argv, &vs, &version_number);
  version_string = std::string(vs);

  if (argc < 4){
    std::cout << "specify data file name, iterations, and block size\n";
    std::cout << "example: ./euler3d ../../data/cfd/fvcorr.domn.097K 1000 16\n";
    return 0;
  }
  const char* data_file_name = argv[1];
  int iterations = atoi(argv[2]);
  block_size = atoi(argv[3]);
  //_clCmdParams(argc, argv);
  cl_mem ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,ff_flux_contribution_momentum_z,  ff_flux_contribution_density_energy;
  cl_mem areas = NULL, elements_surrounding_elements = NULL, normals = NULL;
  cl_mem variables = NULL, old_variables = NULL, fluxes = NULL,
      step_factors = NULL;
  float h_ff_variable[NVAR];
#if 0  
  cl_mem flux_contribution_momentum_x = NULL,
      flux_contribution_momentum_y = NULL,
      flux_contribution_momentum_z = NULL,
      flux_contribution_density_energy = NULL;
#endif  

  //_clInit(device_type, device_id);
  _clInit();

  load_kernels(KERNEL_PREFIX);
  
  // set far field conditions and load them into constant memory on the gpu
  {
    //float h_ff_variable[NVAR];
    const float angle_of_attack = float(3.1415926535897931 / 180.0f) * float(deg_angle_of_attack);
			
    h_ff_variable[VAR_DENSITY] = float(1.4);
			
    float ff_pressure = float(1.0f);
    float ff_speed_of_sound = sqrt(GAMMA*ff_pressure / h_ff_variable[VAR_DENSITY]);
    float ff_speed = float(ff_mach)*ff_speed_of_sound;
			
    FLOAT3 ff_velocity;
    ff_velocity.x = ff_speed*float(cos((float)angle_of_attack));
    ff_velocity.y = ff_speed*float(sin((float)angle_of_attack));
    ff_velocity.z = 0.0f;
			
    h_ff_variable[VAR_MOMENTUM+0] = h_ff_variable[VAR_DENSITY] * ff_velocity.x;
    h_ff_variable[VAR_MOMENTUM+1] = h_ff_variable[VAR_DENSITY] * ff_velocity.y;
    h_ff_variable[VAR_MOMENTUM+2] = h_ff_variable[VAR_DENSITY] * ff_velocity.z;
					
    h_ff_variable[VAR_DENSITY_ENERGY] = h_ff_variable[VAR_DENSITY]*(float(0.5f)*(ff_speed*ff_speed)) + (ff_pressure / float(GAMMA-1.0f));

    FLOAT3 h_ff_momentum;
    h_ff_momentum.x = *(h_ff_variable+VAR_MOMENTUM+0);
    h_ff_momentum.y = *(h_ff_variable+VAR_MOMENTUM+1);
    h_ff_momentum.z = *(h_ff_variable+VAR_MOMENTUM+2);
    FLOAT3 h_ff_flux_contribution_momentum_x;
    FLOAT3 h_ff_flux_contribution_momentum_y;
    FLOAT3 h_ff_flux_contribution_momentum_z;
    FLOAT3 h_ff_flux_contribution_density_energy;
    compute_flux_contribution(h_ff_variable[VAR_DENSITY], h_ff_momentum, h_ff_variable[VAR_DENSITY_ENERGY], ff_pressure, ff_velocity, h_ff_flux_contribution_momentum_x, h_ff_flux_contribution_momentum_y, h_ff_flux_contribution_momentum_z, h_ff_flux_contribution_density_energy);

    // copy far field conditions to the gpu
    //cl_mem ff_variable, ff_flux_contribution_momentum_x, ff_flux_contribution_momentum_y,ff_flux_contribution_momentum_z,  ff_flux_contribution_density_energy;
    ff_variable = _clMalloc(NVAR*sizeof(float));
    ff_flux_contribution_momentum_x = _clMalloc(sizeof(FLOAT3));
    ff_flux_contribution_momentum_y = _clMalloc(sizeof(FLOAT3));
    ff_flux_contribution_momentum_z = _clMalloc(sizeof(FLOAT3));
    ff_flux_contribution_density_energy = _clMalloc(sizeof(FLOAT3));
    _clMemcpyH2D(ff_variable,          h_ff_variable,          NVAR*sizeof(float));
    _clMemcpyH2D(ff_flux_contribution_momentum_x, &h_ff_flux_contribution_momentum_x, sizeof(FLOAT3));
    _clMemcpyH2D(ff_flux_contribution_momentum_y, &h_ff_flux_contribution_momentum_y, sizeof(FLOAT3));
    _clMemcpyH2D(ff_flux_contribution_momentum_z, &h_ff_flux_contribution_momentum_z, sizeof(FLOAT3));		
    _clMemcpyH2D(ff_flux_contribution_density_energy, &h_ff_flux_contribution_density_energy, sizeof(FLOAT3));
    _clFinish();
  }
  
  int nel;
  int nelr;
  // read in domain geometry
  //float* areas;
  //int* elements_surrounding_elements;
  //float* normals;
  {
    std::ifstream file(data_file_name);
    if(!file.is_open()){
      std::cerr << "can not find/open file!\n";
      abort();
    }
    file >> nel;
    nelr = block_size*((nel / block_size )+ std::min(1, nel % block_size));
    std::cout<<"--cambine: nel="<<nel<<", nelr="<<nelr<<std::endl;
    float* h_areas = new float[nelr];
    int* h_elements_surrounding_elements = new int[nelr*NNB];
    float* h_normals = new float[nelr*NDIM*NNB];
					
    // read in data
    for(int i = 0; i < nel; i++)
    {
      file >> h_areas[i];
      for(int j = 0; j < NNB; j++)
      {
        file >> h_elements_surrounding_elements[i + j*nelr];
        if(h_elements_surrounding_elements[i+j*nelr] < 0) h_elements_surrounding_elements[i+j*nelr] = -1;
        h_elements_surrounding_elements[i + j*nelr]--; //it's coming in with Fortran numbering				
					
        for(int k = 0; k < NDIM; k++)
        {
          file >> h_normals[i + (j + k*NNB)*nelr];
          h_normals[i + (j + k*NNB)*nelr] = -h_normals[i + (j + k*NNB)*nelr];
        }
      }
    }
			
    // fill in remaining data
    int last = nel-1;
    for(int i = nel; i < nelr; i++)
    {
      h_areas[i] = h_areas[last];
      for(int j = 0; j < NNB; j++)
      {
        // duplicate the last element
        h_elements_surrounding_elements[i + j*nelr] = h_elements_surrounding_elements[last + j*nelr];	
        for(int k = 0; k < NDIM; k++) h_normals[last + (j + k*NNB)*nelr] = h_normals[last + (j + k*NNB)*nelr];
      }
    }

    areas = alloc<float>(nelr);
    upload<float>(areas, h_areas, nelr);

    elements_surrounding_elements = alloc<int>(nelr*NNB);
    upload<int>(elements_surrounding_elements, h_elements_surrounding_elements, nelr*NNB);

    normals = alloc<float>(nelr*NDIM*NNB);
    upload<float>(normals, h_normals, nelr*NDIM*NNB);

    delete[] h_areas;
    delete[] h_elements_surrounding_elements;
    delete[] h_normals;
  }

  // Create arrays and set initial conditions
  float hv[nelr*NVAR];
  fill_device_variable(nelr, h_ff_variable, hv);
  variables = alloc<float>(nelr*NVAR);				
  upload(variables, hv, nelr*NVAR);
  old_variables = alloc<float>(nelr*NVAR);   	
  fluxes = alloc<float>(nelr*NVAR);
  step_factors = alloc<float>(nelr); 
  // make sure all memory is floatly allocated before we start timing
  upload(old_variables, hv, nelr*NVAR);
  upload(fluxes, hv, nelr*NVAR);  
  _clMemset(step_factors, 0, sizeof(float)*nelr);

  // pre-computed arrays
  cl_mem flux_contribution_momentum_x = alloc<float>(nelr*NDIM);
  cl_mem flux_contribution_momentum_y = alloc<float>(nelr*NDIM);
  cl_mem flux_contribution_momentum_z = alloc<float>(nelr*NDIM);
  cl_mem flux_contribution_density_energy = alloc<float>(nelr*NDIM);
  
  // make sure CUDA isn't still doing something before we start timing
  _clFinish();
  // these need to be computed the first time in order to compute time step
  std::cout << "Starting..." << std::endl;

  TimeStamp start, end;
  GetTime(start);
          
  // Begin iterations
  for(int i = 0; i < iterations; i++){
    copy<float>(old_variables, variables, nelr*NVAR);
    // for the first iteration we compute the time step
    compute_step_factor(nelr, variables, areas, step_factors);
    for(int j = 0; j < RK; j++){
      compute_flux_contributions(nelr, variables,
                                 flux_contribution_momentum_x,
                                 flux_contribution_momentum_y,
                                 flux_contribution_momentum_z,
                                 flux_contribution_density_energy);
      compute_flux(nelr, elements_surrounding_elements, normals,
                   variables, ff_variable, fluxes,
                   ff_flux_contribution_density_energy,
                   ff_flux_contribution_momentum_x,
                   ff_flux_contribution_momentum_y,
                   ff_flux_contribution_momentum_z,
                   flux_contribution_momentum_x,
                   flux_contribution_momentum_y,
                   flux_contribution_momentum_z,
                   flux_contribution_density_energy);
      time_step(j, nelr, old_variables, variables, step_factors, fluxes);
    }
  }
  

  _clFinish();
  GetTime(end);

  printf("Computation done in %0.3lf ms.\n", TimeDiff(start, end));
  
  std::cout << "Saving solution..." << std::endl;
  dump(variables, nel, nelr);
  std::cout << "Saved solution..." << std::endl;
  _clStatistics();
  std::cout << "Cleaning up..." << std::endl;

  //--release resources
  _clFree(ff_variable);
  _clFree(ff_flux_contribution_momentum_x);
  _clFree(ff_flux_contribution_momentum_y);
  _clFree(ff_flux_contribution_momentum_z);
  _clFree(ff_flux_contribution_density_energy);
  _clFree(areas);
  _clFree(elements_surrounding_elements);
  _clFree(normals);
  _clFree(variables);
  _clFree(old_variables);
  _clFree(fluxes);
  _clFree(step_factors);
  _clRelease();
  std::cout << "Done..." << std::endl;

  return 0;
}
