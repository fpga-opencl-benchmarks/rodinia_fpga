#include "host_common.h"

std::vector<cl_kernel> kernels;
std::vector<std::string> kernel_names;
std::string version_string;
int version_number;
int block_size;

int compute_step_factor_idx;
int time_step_idx;
int compute_flux_contributions_idx;
int compute_flux_idx;

void dump(cl_mem variables, int nel, int nelr){
  float* h_variables = new float[nelr*NVAR];
  download(h_variables, variables, nelr*NVAR);

  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY*nelr] << std::endl;
  }

  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++)
    {
      for(int j = 0; j != NDIM; j++)
        file << h_variables[i + (VAR_MOMENTUM+j)*nelr] << " ";
      file << std::endl;
    }
  }
	
  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables[i + VAR_DENSITY_ENERGY*nelr] << std::endl;
  }
  delete[] h_variables;
}

void dump(VariablesD variables, int nel, int nelr){
  VariablesH h_variables(nelr);
  download(h_variables, variables);

  {
    std::ofstream file("density");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables.density[i] << std::endl;
  }

  {
    std::ofstream file("momentum");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++)
    {
      file << h_variables.momentum_x[i] << " ";
      file << h_variables.momentum_y[i] << " ";
      file << h_variables.momentum_z[i] << " ";      
      file << std::endl;
    }
  }
	
  {
    std::ofstream file("density_energy");
    file << nel << " " << nelr << std::endl;
    for(int i = 0; i < nel; i++) file << h_variables.energy[i] << std::endl;
  }
}


void fill_device_variable(int nelr, const float *ff_variable,
                          float *d_variable) {
  for (int i = 0; i < nelr; ++i) {
    for (int j = 0; j < NVAR; ++j) {
      d_variable[i + j*nelr] = ff_variable[j];      
    }
  }
}

void fill_device_variable(int nelr, const float *ff_variable,
                          VariablesH &d_variables) {
  for (int i = 0; i < nelr; ++i) {
    d_variables.density[i] = ff_variable[VAR_DENSITY];
    d_variables.momentum_x[i] = ff_variable[VAR_MOMENTUM + 0];
    d_variables.momentum_y[i] = ff_variable[VAR_MOMENTUM + 1];
    d_variables.momentum_z[i] = ff_variable[VAR_MOMENTUM + 2];
    d_variables.energy[i] = ff_variable[VAR_DENSITY_ENERGY];
  }
}


static bool compute_step_factor_arg_set = false;

void compute_step_factor(int nelr, cl_mem variables, cl_mem areas, cl_mem step_factors){

  int work_items = nelr;
  int work_group_size = block_size;
  cl_kernel kernel = kernels[compute_step_factor_idx];
  if (!compute_step_factor_arg_set) {
    int arg_idx = 0;
    _clSetArgs(kernel, arg_idx++, variables);
    _clSetArgs(kernel, arg_idx++, areas);
    _clSetArgs(kernel, arg_idx++, step_factors);
    _clSetArgs(kernel, arg_idx++, &nelr, sizeof(int));
    compute_step_factor_arg_set = true;
  }
  if (is_ndrange_kernel(version_number)) {
    _clInvokeKernel(kernel, work_items, work_group_size);
  } else {
    _clInvokeKernel(kernel);
  }
}

void compute_step_factor(int nelr, VariablesD &variables, cl_mem areas, cl_mem step_factors){

  int work_items = nelr;
  int work_group_size = block_size;
  cl_kernel kernel = kernels[compute_step_factor_idx];
  if (!compute_step_factor_arg_set) {
    int arg_idx = 0;
    _clSetArgs(kernel, arg_idx++, variables.density);
    _clSetArgs(kernel, arg_idx++, variables.momentum_x);
    _clSetArgs(kernel, arg_idx++, variables.momentum_y);
    _clSetArgs(kernel, arg_idx++, variables.momentum_z);
    _clSetArgs(kernel, arg_idx++, variables.energy);    
    _clSetArgs(kernel, arg_idx++, areas);
    _clSetArgs(kernel, arg_idx++, step_factors);
    _clSetArgs(kernel, arg_idx++, &nelr, sizeof(int));
    compute_step_factor_arg_set = true;
  }
  if (is_ndrange_kernel(version_number)) {
    _clInvokeKernel(kernel, work_items, work_group_size);
  } else {
    _clInvokeKernel(kernel);
  }
}


static bool time_step_arg_set = false;

void time_step(int j, int nelr, cl_mem old_variables,
               cl_mem variables, cl_mem step_factors, cl_mem fluxes){

  int work_items = nelr;
  int work_group_size = block_size;
  cl_kernel kernel = kernels[time_step_idx];
  int arg_idx = 0;
  _clSetArgs(kernel, arg_idx++, &j, sizeof(int));
  if (!time_step_arg_set) {
    _clSetArgs(kernel, arg_idx++, &nelr, sizeof(int));
    _clSetArgs(kernel, arg_idx++, old_variables);
    _clSetArgs(kernel, arg_idx++, variables);
    _clSetArgs(kernel, arg_idx++, step_factors);
    _clSetArgs(kernel, arg_idx++, fluxes);
    time_step_arg_set = true;
  }
  if (is_ndrange_kernel(version_number)) {
    _clInvokeKernel(kernel, work_items, work_group_size);
  } else {
    _clInvokeKernel(kernel);
  }
}

void time_step(int j, int nelr, VariablesD &old_variables,
               VariablesD &variables, cl_mem step_factors, VariablesD &fluxes){

  int work_items = nelr;
  int work_group_size = block_size;
  cl_kernel kernel = kernels[time_step_idx];
  int arg_idx = 0;
  _clSetArgs(kernel, arg_idx++, &j, sizeof(int));
  if (!time_step_arg_set) {
    _clSetArgs(kernel, arg_idx++, &nelr, sizeof(int));
    _clSetArgs(kernel, arg_idx++, old_variables.density);
    _clSetArgs(kernel, arg_idx++, old_variables.momentum_x);
    _clSetArgs(kernel, arg_idx++, old_variables.momentum_y);
    _clSetArgs(kernel, arg_idx++, old_variables.momentum_z);
    _clSetArgs(kernel, arg_idx++, old_variables.energy);
    _clSetArgs(kernel, arg_idx++, variables.density);
    _clSetArgs(kernel, arg_idx++, variables.momentum_x);
    _clSetArgs(kernel, arg_idx++, variables.momentum_y);
    _clSetArgs(kernel, arg_idx++, variables.momentum_z);
    _clSetArgs(kernel, arg_idx++, variables.energy);
    _clSetArgs(kernel, arg_idx++, step_factors);
    _clSetArgs(kernel, arg_idx++, fluxes.density);
    _clSetArgs(kernel, arg_idx++, fluxes.momentum_x);
    _clSetArgs(kernel, arg_idx++, fluxes.momentum_y);
    _clSetArgs(kernel, arg_idx++, fluxes.momentum_z);
    _clSetArgs(kernel, arg_idx++, fluxes.energy);
    time_step_arg_set = true;
  }
  if (is_ndrange_kernel(version_number)) {
    _clInvokeKernel(kernel, work_items, work_group_size);
  } else {
    _clInvokeKernel(kernel);
  }
}


void compute_flux_contribution(float& density, FLOAT3& momentum, float& density_energy, float& pressure, FLOAT3& velocity, FLOAT3& fc_momentum_x, FLOAT3& fc_momentum_y, FLOAT3& fc_momentum_z, FLOAT3& fc_density_energy)
{
  fc_momentum_x.x = velocity.x*momentum.x + pressure;
  fc_momentum_x.y = velocity.x*momentum.y;
  fc_momentum_x.z = velocity.x*momentum.z;
	
	
  fc_momentum_y.x = fc_momentum_x.y;
  fc_momentum_y.y = velocity.y*momentum.y + pressure;
  fc_momentum_y.z = velocity.y*momentum.z;

  fc_momentum_z.x = fc_momentum_x.z;
  fc_momentum_z.y = fc_momentum_y.z;
  fc_momentum_z.z = velocity.z*momentum.z + pressure;

  float de_p = density_energy+pressure;
  fc_density_energy.x = velocity.x*de_p;
  fc_density_energy.y = velocity.y*de_p;
  fc_density_energy.z = velocity.z*de_p;
}

void load_kernels(const std::string &kernel_prefix) {
  char *kernel_file_path = getVersionedKernelName2(kernel_prefix.c_str(),
                                                   version_string.c_str());
  size_t sourcesize;
  char *source = read_kernel(kernel_file_path, &sourcesize);
  
  // compile kernel
  cl_int err = 0;
#ifdef USE_JIT
  const char * slist[2] = { source, 0 };
  cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
#else
  cl_program prog = clCreateProgramWithBinary(context, 1, device_list,
                                              &sourcesize, (const unsigned char**)&source, NULL, &err);
#endif
  if(err != CL_SUCCESS) {
    printf("ERROR: clCreateProgramWithSource/Binary() => %d\n", err);
    display_error_message(err, stderr);
    exit(1);
  }

  char clOptions[110];
  sprintf(clOptions, "-I.");

#ifdef USE_JIT
#ifdef USE_RESTRICT
  sprintf(clOptions + strlen(clOptions), " -DUSE_RESTRICT");
#endif  
  sprintf(clOptions + strlen(clOptions), " -DBSIZE=%d", block_size);
#endif
  printf("kernel compile options: %s\n", clOptions);
  
  clBuildProgram_SAFE(prog, num_devices, device_list, clOptions, NULL, NULL);
  
  for (unsigned nKernel = 0; nKernel < kernel_names.size(); nKernel++) {
    // get a kernel object handle for a kernel with the given name
    cl_int err;
    cl_kernel kernel = clCreateKernel(prog,
                                      (kernel_names[nKernel]).c_str(),
                                      &err);
    if ((err != CL_SUCCESS) || (kernel == NULL)) {
      std::cerr << "InitCL()::Error: Creating Kernel (clCreateKernel) \""
                << kernel_names[nKernel] << "\"";
      exit(1);
    }

    kernels.push_back(kernel);
  }
}

