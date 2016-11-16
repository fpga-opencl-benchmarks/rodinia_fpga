/* ============================================================
//--functions: 	kernel funtion
//--programmer:	Jianbin Fang
//--date:		24/03/2011
============================================================ */
#include "common.h"
#include "kernel_common.h"
#include "../common/opencl_kernel_common.h"

__kernel void compute_step_factor(__global float* variables, 
                                  __global float* areas, 
                                  __global float* step_factors,
                                  int nelr){
  //const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  const int i = get_global_id(0);
  if( i >= nelr) return;

  float density = variables[i + VAR_DENSITY*nelr];
  FLOAT3 momentum;
  momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
  momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
  momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
  float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
  FLOAT3 velocity;       compute_velocity(density, momentum, &velocity);
  float speed_sqd      = compute_speed_sqd(velocity);
  //float speed_sqd;
  //compute_speed_sqd(velocity, speed_sqd);
  float pressure       = compute_pressure(density, density_energy, speed_sqd);
  float speed_of_sound = compute_speed_of_sound(density, pressure);

  // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
  //step_factors[i] = (float)(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
  step_factors[i] = (float)(0.5f) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
}

__kernel void compute_flux(
    __global int* elements_surrounding_elements, 
    __global float* normals, 
    __global float* variables, 
    __constant float* ff_variable,
    __global float* fluxes,
    __constant FLOAT3* ff_flux_contribution_density_energy,
    __constant FLOAT3* ff_flux_contribution_momentum_x,
    __constant FLOAT3* ff_flux_contribution_momentum_y,
    __constant FLOAT3* ff_flux_contribution_momentum_z,
    int nelr){
  const float smoothing_coefficient = (float)(0.2f);
  //const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  const int i = get_global_id(0);
  if( i >= nelr) return;
  int j, nb;
  FLOAT3 normal; float normal_len;
  float factor;
	
  float density_i = variables[i + VAR_DENSITY*nelr];
  FLOAT3 momentum_i;
  momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
  momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
  momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

  float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

  FLOAT3 velocity_i;             				compute_velocity(density_i, momentum_i, &velocity_i);
  float speed_sqd_i                          = compute_speed_sqd(velocity_i);
  //float speed_sqd_i;
  //compute_speed_sqd(velocity_i, speed_sqd_i);
  //float speed_i                              = sqrtf(speed_sqd_i);
  float speed_i                              = sqrt(speed_sqd_i);
  float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
  float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
  FLOAT3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
  FLOAT3 flux_contribution_i_density_energy;	
  compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, &flux_contribution_i_momentum_x, &flux_contribution_i_momentum_y, &flux_contribution_i_momentum_z, &flux_contribution_i_density_energy);
	
  float flux_i_density = (float)(0.0f);
  FLOAT3 flux_i_momentum;
  flux_i_momentum.x = (float)(0.0f);
  flux_i_momentum.y = (float)(0.0f);
  flux_i_momentum.z = (float)(0.0f);
  float flux_i_density_energy = (float)(0.0f);
		
  FLOAT3 velocity_nb;
  float density_nb, density_energy_nb;
  FLOAT3 momentum_nb;
  FLOAT3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
  FLOAT3 flux_contribution_nb_density_energy;	
  float speed_sqd_nb, speed_of_sound_nb, pressure_nb;

  // Logic not sufficient for de5net_a7
  //#pragma unroll
  for(j = 0; j < NNB; j++)
  {
    nb = elements_surrounding_elements[i + j*nelr];
    normal.x = normals[i + (j + 0*NNB)*nelr];
    normal.y = normals[i + (j + 1*NNB)*nelr];
    normal.z = normals[i + (j + 2*NNB)*nelr];
    //normal_len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
    normal_len = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		
    if(nb >= 0) 	// a legitimate neighbor
    {
      density_nb = variables[nb + VAR_DENSITY*nelr];
      momentum_nb.x = variables[nb + (VAR_MOMENTUM+0)*nelr];
      momentum_nb.y = variables[nb + (VAR_MOMENTUM+1)*nelr];
      momentum_nb.z = variables[nb + (VAR_MOMENTUM+2)*nelr];
      density_energy_nb = variables[nb + VAR_DENSITY_ENERGY*nelr];
      compute_velocity(density_nb, momentum_nb, &velocity_nb);
      speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
      pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
      speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
      compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, &flux_contribution_nb_momentum_x, &flux_contribution_nb_momentum_y, &flux_contribution_nb_momentum_z, &flux_contribution_nb_density_energy);
			
      // artificial viscosity
      factor = -normal_len*smoothing_coefficient*(float)(0.5f)*(speed_i + sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
      flux_i_density += factor*(density_i-density_nb);
      flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
      flux_i_momentum.x += factor*(momentum_i.x-momentum_nb.x);
      flux_i_momentum.y += factor*(momentum_i.y-momentum_nb.y);
      flux_i_momentum.z += factor*(momentum_i.z-momentum_nb.z);

      // accumulate cell-centered fluxes
      factor = (float)(0.5f)*normal.x;
      flux_i_density += factor*(momentum_nb.x+momentum_i.x);
      flux_i_density_energy += factor*(flux_contribution_nb_density_energy.x+flux_contribution_i_density_energy.x);
      flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.x+flux_contribution_i_momentum_x.x);
      flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.x+flux_contribution_i_momentum_y.x);
      flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.x+flux_contribution_i_momentum_z.x);
			
      factor = (float)(0.5f)*normal.y;
      flux_i_density += factor*(momentum_nb.y+momentum_i.y);
      flux_i_density_energy += factor*(flux_contribution_nb_density_energy.y+flux_contribution_i_density_energy.y);
      flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.y+flux_contribution_i_momentum_x.y);
      flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.y+flux_contribution_i_momentum_y.y);
      flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.y+flux_contribution_i_momentum_z.y);
			
      factor = (float)(0.5f)*normal.z;
      flux_i_density += factor*(momentum_nb.z+momentum_i.z);
      flux_i_density_energy += factor*(flux_contribution_nb_density_energy.z+flux_contribution_i_density_energy.z);
      flux_i_momentum.x += factor*(flux_contribution_nb_momentum_x.z+flux_contribution_i_momentum_x.z);
      flux_i_momentum.y += factor*(flux_contribution_nb_momentum_y.z+flux_contribution_i_momentum_y.z);
      flux_i_momentum.z += factor*(flux_contribution_nb_momentum_z.z+flux_contribution_i_momentum_z.z);
    }
    else if(nb == -1)	// a wing boundary
    {
      flux_i_momentum.x += normal.x*pressure_i;
      flux_i_momentum.y += normal.y*pressure_i;
      flux_i_momentum.z += normal.z*pressure_i;
    }
    else if(nb == -2) // a far field boundary
    {
      factor = (float)(0.5f)*normal.x;
      flux_i_density += factor*(ff_variable[VAR_MOMENTUM+0]+momentum_i.x);
      flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].x+flux_contribution_i_density_energy.x);
      flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].x + flux_contribution_i_momentum_x.x);
      flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].x + flux_contribution_i_momentum_y.x);
      flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].x + flux_contribution_i_momentum_z.x);
			
      factor = (float)(0.5f)*normal.y;
      flux_i_density += factor*(ff_variable[VAR_MOMENTUM+1]+momentum_i.y);
      flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].y+flux_contribution_i_density_energy.y);
      flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].y + flux_contribution_i_momentum_x.y);
      flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].y + flux_contribution_i_momentum_y.y);
      flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].y + flux_contribution_i_momentum_z.y);

      factor = (float)(0.5f)*normal.z;
      flux_i_density += factor*(ff_variable[VAR_MOMENTUM+2]+momentum_i.z);
      flux_i_density_energy += factor*(ff_flux_contribution_density_energy[0].z+flux_contribution_i_density_energy.z);
      flux_i_momentum.x += factor*(ff_flux_contribution_momentum_x[0].z + flux_contribution_i_momentum_x.z);
      flux_i_momentum.y += factor*(ff_flux_contribution_momentum_y[0].z + flux_contribution_i_momentum_y.z);
      flux_i_momentum.z += factor*(ff_flux_contribution_momentum_z[0].z + flux_contribution_i_momentum_z.z);

    }
  }

  fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
  fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
  fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
  fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
  fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}

__kernel void time_step(int j, int nelr, 
                        __global float* old_variables, 
                        __global float* variables, 
                        __global float* step_factors, 
                        __global float* fluxes){
  //const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  const int i = get_global_id(0);
  if( i >= nelr) return;

  float factor = step_factors[i]/(float)(RK+1-j);

  variables[i + VAR_DENSITY*nelr] = old_variables[i + VAR_DENSITY*nelr] + factor*fluxes[i + VAR_DENSITY*nelr];
  variables[i + VAR_DENSITY_ENERGY*nelr] = old_variables[i + VAR_DENSITY_ENERGY*nelr] + factor*fluxes[i + VAR_DENSITY_ENERGY*nelr];
  variables[i + (VAR_MOMENTUM+0)*nelr] = old_variables[i + (VAR_MOMENTUM+0)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+0)*nelr];
  variables[i + (VAR_MOMENTUM+1)*nelr] = old_variables[i + (VAR_MOMENTUM+1)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+1)*nelr];	
  variables[i + (VAR_MOMENTUM+2)*nelr] = old_variables[i + (VAR_MOMENTUM+2)*nelr] + factor*fluxes[i + (VAR_MOMENTUM+2)*nelr];	
	
}


