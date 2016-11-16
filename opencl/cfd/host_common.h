#ifndef HOST_COMMON_H_
#define HOST_COMMON_H_

#include "CLHelper2.h"

#include <vector>
#include <iostream>
#include <fstream>

#include "common.h"

extern std::vector<cl_kernel> kernels;
extern std::vector<std::string> kernel_names;
extern std::string version_string;
extern int version_number;
extern int block_size;

extern int compute_step_factor_idx;
extern int time_step_idx;
extern int compute_flux_contributions_idx;
extern int compute_flux_idx;

/*
 * Generic functions
 */
template <typename T>
cl_mem alloc(int N){
  cl_mem mem_d = _clMalloc(sizeof(T)*N);
  return mem_d;
}

template <typename T>
void dealloc(cl_mem array){
  _clFree(array);
}

template <typename T>
void copy(cl_mem dst, cl_mem src, int N){
  _clMemcpyD2D(dst, src, N*sizeof(T));
}

template <typename T>
void upload(cl_mem dst, T* src, int N){
  _clMemcpyH2D(dst, src, N*sizeof(T));
}

struct VariablesH {
  float *density;
  float *momentum_x;
  float *momentum_y;
  float *momentum_z;
  float *energy;
  int n;
  VariablesH(int n) {
    density = (float *)alignedMalloc(sizeof(float) * n);
    momentum_x = (float *)alignedMalloc(sizeof(float) * n);
    momentum_y = (float *)alignedMalloc(sizeof(float) * n);
    momentum_z = (float *)alignedMalloc(sizeof(float) * n);
    energy = (float *)alignedMalloc(sizeof(float) * n);
    this->n = n;
  }
  ~VariablesH() {
    free(density);
    free(momentum_x);
    free(momentum_y);
    free(momentum_z);
    free(energy);
  }
};

struct VariablesD {
  cl_mem density;
  cl_mem momentum_x;
  cl_mem momentum_y;
  cl_mem momentum_z;
  cl_mem energy;
  int n;
  VariablesD() {}  
  void allocate(int n) {
    density = alloc<float>(n);
    momentum_x = alloc<float>(n);
    momentum_y = alloc<float>(n);
    momentum_z = alloc<float>(n);
    energy = alloc<float>(n);
    this->n = n;
  }
};

template <typename T>
void copy(VariablesD &dst, VariablesD &src, int N){
  copy<T>(dst.density, src.density, src.n);
  copy<T>(dst.energy, src.energy, src.n);
  copy<T>(dst.momentum_x, src.momentum_x, src.n);
  copy<T>(dst.momentum_y, src.momentum_y, src.n);
  copy<T>(dst.momentum_z, src.momentum_z, src.n);  
}

inline void upload(VariablesD &dst, VariablesH &src) {
  upload(dst.density, src.density, src.n);
  upload(dst.energy, src.energy, src.n);
  upload(dst.momentum_x, src.momentum_x, src.n);
  upload(dst.momentum_y, src.momentum_y, src.n);
  upload(dst.momentum_z, src.momentum_z, src.n);
}

template <typename T>
void download(T* dst, cl_mem src, int N){
  _clMemcpyD2H(dst, src, N*sizeof(T));
}

inline void download(VariablesH &dst, VariablesD &src) {
  download(dst.density, src.density, src.n);
  download(dst.energy, src.energy, src.n);
  download(dst.momentum_x, src.momentum_x, src.n);
  download(dst.momentum_y, src.momentum_y, src.n);
  download(dst.momentum_z, src.momentum_z, src.n);
}


void dump(cl_mem variables, int nel, int nelr);
void dump(VariablesD variables, int nel, int nelr);
void fill_device_variable(int nelr, const float *ff_variable,
                          float *d_variable);
void fill_device_variable(int nelr, const float *ff_variable,
                          VariablesH &d_variables);
void compute_step_factor(int nelr, cl_mem variables,
                         cl_mem areas, cl_mem step_factors);
void compute_step_factor(int nelr, VariablesD &variables,
                         cl_mem areas, cl_mem step_factors);

void time_step(int j, int nelr, cl_mem old_variables,
               cl_mem variables, cl_mem step_factors, cl_mem fluxes);

void time_step(int j, int nelr, VariablesD &old_variables,
               VariablesD &variables, cl_mem step_factors, VariablesD &fluxes);

void compute_flux_contribution(float& density, FLOAT3& momentum,
                               float& density_energy, float& pressure,
                               FLOAT3& velocity, FLOAT3& fc_momentum_x,
                               FLOAT3& fc_momentum_y, FLOAT3& fc_momentum_z,
                               FLOAT3& fc_density_energy);
void load_kernels(const std::string &kernel_prefix);


#endif /* HOST_COMMON_H_ */
