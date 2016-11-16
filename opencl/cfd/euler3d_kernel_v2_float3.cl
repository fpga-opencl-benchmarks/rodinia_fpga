/* ============================================================
//--functions: 	kernel funtion
//--programmer:	Jianbin Fang
//--date:		24/03/2011
============================================================ */
#include "common.h"
#include "../common/opencl_kernel_common.h"

//--cambine: omit &
#if 0
static inline void compute_velocity(float  density, FLOAT3 momentum,
                                    FLOAT3* RESTRICT velocity){
  velocity->x = momentum.x / density;
  velocity->y = momentum.y / density;
  velocity->z = momentum.z / density;
}
#else
static inline void compute_velocity(float  density, float3 momentum,
                                    float3 *velocity){
  *velocity = momentum / density;
}
#endif	

#if 0
static inline float compute_speed_sqd(FLOAT3 velocity){
  return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}
#else
static inline float compute_speed_sqd(float3 velocity){
  return velocity.x*velocity.x + velocity.y*velocity.y + velocity.z*velocity.z;
}
#endif

static inline float compute_pressure(float density, float density_energy,
                                     float speed_sqd){
  return ((float)(GAMMA) - (float)(1.0f))*(density_energy - (float)(0.5f)*density*speed_sqd);
}
static inline float compute_speed_of_sound(float density, float pressure){
  //return sqrtf(float(GAMMA)*pressure/density);
  return sqrt((float)(GAMMA)*pressure/density);
}
#if 0
static inline void compute_flux_contribution(float density, FLOAT3 momentum,
                                             float density_energy, float pressure,
                                             FLOAT3 velocity,
                                             FLOAT3* RESTRICT fc_momentum_x,
                                             FLOAT3* RESTRICT fc_momentum_y,
                                             FLOAT3* RESTRICT fc_momentum_z,
                                             FLOAT3* RESTRICT fc_density_energy)
{
  fc_momentum_x->x = velocity.x*momentum.x + pressure;
  fc_momentum_x->y = velocity.x*momentum.y;
  fc_momentum_x->z = velocity.x*momentum.z;
	
	
  fc_momentum_y->x = fc_momentum_x->y;
  fc_momentum_y->y = velocity.y*momentum.y + pressure;
  fc_momentum_y->z = velocity.y*momentum.z;

  fc_momentum_z->x = fc_momentum_x->z;
  fc_momentum_z->y = fc_momentum_y->z;
  fc_momentum_z->z = velocity.z*momentum.z + pressure;

  float de_p = density_energy+pressure;
  fc_density_energy->x = velocity.x*de_p;
  fc_density_energy->y = velocity.y*de_p;
  fc_density_energy->z = velocity.z*de_p;
}
#else
static inline void compute_flux_contribution(float density, float3 momentum,
                                             float density_energy, float pressure,
                                             float3 velocity,
                                             float3* RESTRICT fc_momentum_x,
                                             float3* RESTRICT fc_momentum_y,
                                             float3* RESTRICT fc_momentum_z,
                                             float3* RESTRICT fc_density_energy)
{
  *fc_momentum_x = velocity.x * momentum;
  fc_momentum_x->x += pressure;
	
  fc_momentum_y->x = fc_momentum_x->y;
  fc_momentum_y->y = velocity.y*momentum.y + pressure;
  fc_momentum_y->z = velocity.y*momentum.z;

  fc_momentum_z->x = fc_momentum_x->z;
  fc_momentum_z->y = fc_momentum_y->z;
  fc_momentum_z->z = velocity.z*momentum.z + pressure;

  float de_p = density_energy+pressure;
  *fc_density_energy = velocity*de_p;
}
#endif

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__kernel void compute_step_factor(__global float* RESTRICT variables, 
                                  __global float* RESTRICT areas, 
                                  __global float* RESTRICT step_factors,
                                  int nelr){
  //const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  const int i = get_global_id(0);
  if( i >= nelr) return;

  float density = variables[i + VAR_DENSITY*nelr];
  float3 momentum;
  momentum.x = variables[i + (VAR_MOMENTUM+0)*nelr];
  momentum.y = variables[i + (VAR_MOMENTUM+1)*nelr];
  momentum.z = variables[i + (VAR_MOMENTUM+2)*nelr];
	
  float density_energy = variables[i + VAR_DENSITY_ENERGY*nelr];
	
  float3 velocity;       compute_velocity(density, momentum, &velocity);
  float speed_sqd      = compute_speed_sqd(velocity);
  //float speed_sqd;
  //compute_speed_sqd(velocity, speed_sqd);
  float pressure       = compute_pressure(density, density_energy, speed_sqd);
  float speed_of_sound = compute_speed_of_sound(density, pressure);

  // dt = float(0.5f) * sqrtf(areas[i]) /  (||v|| + c).... but when we do time stepping, this later would need to be divided by the area, so we just do it all at once
  //step_factors[i] = (float)(0.5f) / (sqrtf(areas[i]) * (sqrtf(speed_sqd) + speed_of_sound));
  step_factors[i] = (float)(0.5f) / (sqrt(areas[i]) * (sqrt(speed_sqd) + speed_of_sound));
}

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__kernel void compute_flux(
    __global int* RESTRICT elements_surrounding_elements, 
    __global float* RESTRICT normals, 
    __global float* RESTRICT variables, 
    __constant float* RESTRICT ff_variable,
    __global float* RESTRICT fluxes,
    __constant FLOAT3* RESTRICT ff_flux_contribution_density_energy,
    __constant FLOAT3* RESTRICT ff_flux_contribution_momentum_x,
    __constant FLOAT3* RESTRICT ff_flux_contribution_momentum_y,
    __constant FLOAT3* RESTRICT ff_flux_contribution_momentum_z,
    int nelr){
  const float smoothing_coefficient = (float)(0.2f);
  //const int i = (blockDim.x*blockIdx.x + threadIdx.x);
  const int i = get_global_id(0);
  if( i >= nelr) return;
  int j, nb;
  float3 normal; float normal_len;
  float factor;
	
  float density_i = variables[i + VAR_DENSITY*nelr];
  float3 momentum_i;
  momentum_i.x = variables[i + (VAR_MOMENTUM+0)*nelr];
  momentum_i.y = variables[i + (VAR_MOMENTUM+1)*nelr];
  momentum_i.z = variables[i + (VAR_MOMENTUM+2)*nelr];

  float density_energy_i = variables[i + VAR_DENSITY_ENERGY*nelr];

  float3 velocity_i;
  compute_velocity(density_i, momentum_i, &velocity_i);
  float speed_sqd_i                          = compute_speed_sqd(velocity_i);
  //float speed_sqd_i;
  //compute_speed_sqd(velocity_i, speed_sqd_i);
  //float speed_i                              = sqrtf(speed_sqd_i);
  float speed_i                              = sqrt(speed_sqd_i);
  float pressure_i                           = compute_pressure(density_i, density_energy_i, speed_sqd_i);
  float speed_of_sound_i                     = compute_speed_of_sound(density_i, pressure_i);
  float3 flux_contribution_i_momentum_x, flux_contribution_i_momentum_y, flux_contribution_i_momentum_z;
  float3 flux_contribution_i_density_energy;	
  compute_flux_contribution(density_i, momentum_i, density_energy_i, pressure_i, velocity_i, &flux_contribution_i_momentum_x, &flux_contribution_i_momentum_y, &flux_contribution_i_momentum_z, &flux_contribution_i_density_energy);
	
  float flux_i_density = (float)(0.0f);
  float3 flux_i_momentum;
  flux_i_momentum.x = (float)(0.0f);
  flux_i_momentum.y = (float)(0.0f);
  flux_i_momentum.z = (float)(0.0f);
  float flux_i_density_energy = (float)(0.0f);
		
  float3 velocity_nb;
  float density_nb, density_energy_nb;
  float3 momentum_nb;
  float3 flux_contribution_nb_momentum_x, flux_contribution_nb_momentum_y, flux_contribution_nb_momentum_z;
  float3 flux_contribution_nb_density_energy;	
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
      velocity_nb = momentum_nb / density_nb;
      speed_sqd_nb                      = compute_speed_sqd(velocity_nb);
      pressure_nb                       = compute_pressure(density_nb, density_energy_nb, speed_sqd_nb);
      speed_of_sound_nb                 = compute_speed_of_sound(density_nb, pressure_nb);
      compute_flux_contribution(density_nb, momentum_nb, density_energy_nb, pressure_nb, velocity_nb, &flux_contribution_nb_momentum_x, &flux_contribution_nb_momentum_y, &flux_contribution_nb_momentum_z, &flux_contribution_nb_density_energy);
			
      // artificial viscosity
      factor = -normal_len*smoothing_coefficient*(float)(0.5f)*(speed_i + sqrt(speed_sqd_nb) + speed_of_sound_i + speed_of_sound_nb);
      flux_i_density += factor*(density_i-density_nb);
      flux_i_density_energy += factor*(density_energy_i-density_energy_nb);
      flux_i_momentum += factor * (momentum_i - momentum_nb);

      normal *= 0.5f;
      // accumulate cell-centered fluxes
      factor = normal.x;
      float3 fim;

      // flux_i_density
      fim = normal * (momentum_nb + momentum_i);
      flux_i_density += fim.x + fim.y + fim.z;

      // flux_i_density_energy
      fim = normal * (flux_contribution_nb_density_energy +
                      flux_contribution_i_density_energy);
      flux_i_density_energy += fim.x + fim.y + fim.z;

      // flux_i_momentum
      fim = normal * (flux_contribution_nb_momentum_x
                      + flux_contribution_i_momentum_x);
      flux_i_momentum.x += fim.x + fim.y + fim.z;
      fim = normal * (flux_contribution_nb_momentum_y
                      + flux_contribution_i_momentum_y);
      flux_i_momentum.y += fim.x + fim.y + fim.z;
      fim = normal * (flux_contribution_nb_momentum_z
                      + flux_contribution_i_momentum_z);
      flux_i_momentum.z += fim.x + fim.y + fim.z;
    }
    else if(nb == -1)	// a wing boundary
    {
      flux_i_momentum += normal * pressure_i;
    }
    else if(nb == -2) // a far field boundary
    {
      normal *= 0.5f;      
      float3 fim;

      fim.x = ff_variable[VAR_MOMENTUM+0];
      fim.y = ff_variable[VAR_MOMENTUM+1];
      fim.z = ff_variable[VAR_MOMENTUM+2];

      fim = normal * (fim + momentum_i);
      flux_i_density += fim.x + fim.y + fim.z;
      
      fim.x = ff_flux_contribution_density_energy[0].x;
      fim.y = ff_flux_contribution_density_energy[0].y;
      fim.z = ff_flux_contribution_density_energy[0].z;
      fim = normal * (fim + flux_contribution_i_density_energy);
      flux_i_density_energy += fim.x  + fim.y + fim.z;

      fim.x = ff_flux_contribution_momentum_x[0].x;
      fim.y = ff_flux_contribution_momentum_x[0].y;
      fim.z = ff_flux_contribution_momentum_x[0].z;      
      fim = normal * (fim + flux_contribution_i_momentum_x);
      flux_i_momentum.x = fim.x + fim.y + fim.z;
      
      fim.x = ff_flux_contribution_momentum_y[0].x;
      fim.y = ff_flux_contribution_momentum_y[0].y;
      fim.z = ff_flux_contribution_momentum_y[0].z;      
      fim = normal * (fim + flux_contribution_i_momentum_y);
      flux_i_momentum.y = fim.x + fim.y + fim.z;
          
      fim.x = ff_flux_contribution_momentum_z[0].x;
      fim.y = ff_flux_contribution_momentum_z[0].y;
      fim.z = ff_flux_contribution_momentum_z[0].z;      
      fim = normal * (fim + flux_contribution_i_momentum_z);
      flux_i_momentum.z = fim.x + fim.y + fim.z;
    }
  }

  fluxes[i + VAR_DENSITY*nelr] = flux_i_density;
  fluxes[i + (VAR_MOMENTUM+0)*nelr] = flux_i_momentum.x;
  fluxes[i + (VAR_MOMENTUM+1)*nelr] = flux_i_momentum.y;
  fluxes[i + (VAR_MOMENTUM+2)*nelr] = flux_i_momentum.z;
  fluxes[i + VAR_DENSITY_ENERGY*nelr] = flux_i_density_energy;
}

__attribute__((reqd_work_group_size(BSIZE,1,1)))
__kernel void time_step(int j, int nelr, 
                        __global float* RESTRICT old_variables, 
                        __global float* RESTRICT variables, 
                        __global float* RESTRICT step_factors, 
                        __global float* RESTRICT fluxes){
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


