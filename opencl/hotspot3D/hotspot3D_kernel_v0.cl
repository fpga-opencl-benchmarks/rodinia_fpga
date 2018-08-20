#include "hotspot3D_common.h"

__kernel void hotspotOpt1(__global float* restrict p,
                          __global float* restrict tIn,
                          __global float* restrict tOut,
                                   float           sdc,
                                   int             nx,
                                   int             ny,
                                   int             nz,
                                   float           ce,
                                   float           cw, 
                                   float           cn,
                                   float           cs,
                                   float           ct,
                                   float           cb, 
                                   float           cc)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int c = i + j * nx;
  int xy = nx * ny;

  int W = (i == 0)        ? c : c - 1;
  int E = (i == nx-1)     ? c : c + 1;
  int N = (j == 0)        ? c : c - nx;
  int S = (j == ny-1)     ? c : c + nx;

  float temp1, temp2, temp3;
  temp1 = temp2 = tIn[c];
  temp3 = tIn[c+xy];
  tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E]
    + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * AMB_TEMP;
  c += xy;
  W += xy;
  E += xy;
  N += xy;
  S += xy;

  for (int k = 1; k < nz-1; ++k) {
      temp1 = temp2;
      temp2 = temp3;
      temp3 = tIn[c+xy];
      tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E]
        + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * AMB_TEMP;
      c += xy;
      W += xy;
      E += xy;
      N += xy;
      S += xy;
  }
  temp1 = temp2;
  temp2 = temp3;
  tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E]
    + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * AMB_TEMP;
  return;
}


