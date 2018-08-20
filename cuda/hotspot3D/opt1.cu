#include <omp.h>
#include "../../common/timer.h"
#include "../../common/power_gpu.h"

__global__ void hotspotOpt1(float *p, float* tIn, float *tOut, float sdc,
        int nx, int ny, int nz,
        float ce, float cw, 
        float cn, float cs,
        float ct, float cb, 
        float cc) 
{
    float amb_temp = 80.0;

    int i = blockDim.x * blockIdx.x + threadIdx.x;  
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)        ? c : c - 1;
    int E = (i == nx-1)     ? c : c + 1;
    int N = (j == 0)        ? c : c - nx;
    int S = (j == ny-1)     ? c : c + nx;

    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];
    temp3 = tIn[c+xy];
    tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E] + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * amb_temp;
    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    for (int k = 1; k < nz-1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c+xy];
        tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E] + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * amb_temp;
        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cn * tIn[N] + cs * tIn[S] + ce * tIn[E] + cw * tIn[W] + ct * temp3 + cb * temp1 + sdc * p[c] + ct * amb_temp;
    return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut,
        int nx, int ny, int nz,
        float Cap, 
        float Rx, float Ry, float Rz, 
        float dt, int numiter) 
{
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw =stepDivCap/ Rx;
    cn = cs =stepDivCap/ Ry;
    ct = cb =stepDivCap/ Rz;

    TimeStamp start, end;
    double totalTime, power = 0, energy = 0;
    int flag = 0;

    cc = 1.0 - (2.0*ce + 2.0*cn + 3.0*ct);

    size_t s = sizeof(float) * nx * ny * nz;  
    float  *tIn_d, *tOut_d, *p_d;
    cudaMalloc((void**)&p_d,s);
    cudaMalloc((void**)&tIn_d,s);
    cudaMalloc((void**)&tOut_d,s);
    cudaMemcpy(tIn_d, tIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);

    #pragma omp parallel num_threads(2) shared(flag)
    {
        if (omp_get_thread_num() == 0)
        {
            power = GetPowerGPU(&flag, 0);
        }
        else
        {
            #pragma omp barrier
            GetTime(start);
            for (int i = 0; i < numiter; ++i)
            {
                hotspotOpt1<<<grid_dim, block_dim>>>
                    (p_d, tIn_d, tOut_d, stepDivCap, nx, ny, nz, ce, cw, cn, cs, ct, cb, cc);
                float *t = tIn_d;
                tIn_d = tOut_d;
                tOut_d = t;
            }
            cudaDeviceSynchronize();
            GetTime(end);
            flag = 1;
        }
    }

    // output pointer is always swapped one extra time and hence, tIn_d will point to the correct output 
    cudaMemcpy(tOut, tIn_d, s, cudaMemcpyDeviceToHost);

    cudaFree(p_d);
    cudaFree(tIn_d);
    cudaFree(tOut_d);

    totalTime = TimeDiff(start, end);
    energy = GetEnergyGPU(power, totalTime);
    printf("\nComputation done in %0.3lf ms.\n", totalTime);
    printf("Throughput is %0.3lf GBps.\n", (3 * nx * ny * nz * sizeof(float) * numiter) / (1000000000.0 * totalTime / 1000.0));
    if (power != -1) // -1 --> failed to read energy values
    {
        printf("Total energy used is %0.3lf jouls.\n", energy);
        printf("Average power consumption is %0.3lf watts.\n", power);
    }

    return;
}

