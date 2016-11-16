#ifndef COMMON_H_
#define COMMON_H_

#define GAMMA (1.4f)

#define NDIM (3)
#define NNB (4)

#define RK (3)	// 3rd order RK
#define ff_mach (1.2f)
#define deg_angle_of_attack (0.0f)

#define VAR_DENSITY (0)
#define VAR_MOMENTUM  (1)
#define VAR_DENSITY_ENERGY (VAR_MOMENTUM+NDIM)
#define NVAR (VAR_DENSITY_ENERGY+1)
//#pragma OPENCL EXTENSION CL_MAD : enable

//self-defined user type
typedef struct{
  float x;
  float y;
  float z;
} FLOAT3;

#define FLOAT3_ASSIGN(f3, rx, ry, rz) do { \
    (f3).x = rx;                           \
    (f3).y = ry;                           \
    (f3).z = rz;                           \
  } while (0)

#endif /* COMMON_H_ */
