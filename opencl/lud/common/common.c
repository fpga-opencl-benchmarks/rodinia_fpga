#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "common.h"
#ifdef USE_AOT
	#include "../../common/opencl_util.h"
#endif

func_ret_t 
create_matrix_from_file(float **mp, const char* filename, int *size_p){
  int i, j, size;
  float *m;
  FILE *fp = NULL;

  fp = fopen(filename, "rb");
  if ( fp == NULL) {
      return RET_FAILURE;
  }

  fscanf(fp, "%d\n", &size);

#ifdef USE_AOT
  m = (float*) alignedMalloc(sizeof(float)*size*size);
#else
  m = (float*) malloc(sizeof(float)*size*size);
#endif
  
  if ( m == NULL) {
      fclose(fp);
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          fscanf(fp, "%f ", m+i*size+j);
      }
  }

  fclose(fp);

  *size_p = size;
  *mp = m;

  return RET_SUCCESS;
}

// This function is broken since the variable m is not initialized before
// use. The function actually does not seem to be used.
#if 0
func_ret_t
create_matrix_from_random(float **mp, int size){
  float *l, *u, *m;
  int i,j,k;

  srand(time(NULL));

  l = (float*)malloc(size*size*sizeof(float));
  if ( l == NULL)
    return RET_FAILURE;

  u = (float*)malloc(size*size*sizeof(float));
  if ( u == NULL) {
      free(l);
      return RET_FAILURE;
  }

  for (i = 0; i < size; i++) {
      for (j=0; j < size; j++) {
          if (i>j) {
              l[i*size+j] = GET_RAND_FP;
          } else if (i == j) {
              l[i*size+j] = 1;
          } else {
              l[i*size+j] = 0;
          }
      }
  }

  for (j=0; j < size; j++) {
      for (i=0; i < size; i++) {
          if (i>j) {
              u[j*size+i] = 0;
          }else {
              u[j*size+i] = GET_RAND_FP; 
          }
      }
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
          for (k=0; k <= MIN(i,j); k++)
            m[i*size+j] = l[i*size+k] * u[j*size+k];
      }
  }

  free(l);
  free(u);

  *mp = m;

  return RET_SUCCESS;
}
#endif

void
matrix_multiply(float *inputa, float *inputb, float *output, int size){
  int i, j, k;

  for (i=0; i < size; i++)
    for (k=0; k < size; k++)
      for (j=0; j < size; j++)
        output[i*size+j] = inputa[i*size+k] * inputb[k*size+j];

}

func_ret_t
lud_verify(float *m, float *lu, int matrix_dim){
  int i, j, k, verify = 1;
  float *tmp = (float*)malloc(matrix_dim*matrix_dim*sizeof(float));

  for (i=0; i < matrix_dim; i ++)
    for (j=0; j< matrix_dim; j++) {
        float sum = 0;
        float l,u;
        for (k=0; k <= MIN(i,j); k++){
            if ( i==k)
              l=1;
            else
              l=lu[i*matrix_dim+k];
            u=lu[k*matrix_dim+j];
            sum+=l*u;
        }
        tmp[i*matrix_dim+j] = sum;
    }
  /* printf(">>>>>LU<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", lu[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf(">>>>>result<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", tmp[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */
  /* printf(">>>>>input<<<<<<<\n"); */
  /* for (i=0; i<matrix_dim; i++){ */
  /*   for (j=0; j<matrix_dim;j++){ */
  /*       printf("%f ", m[i*matrix_dim+j]); */
  /*   } */
  /*   printf("\n"); */
  /* } */

  for (i=0; i<matrix_dim; i++){
      for (j=0; j<matrix_dim; j++){
          if ( fabs(m[i*matrix_dim+j]-tmp[i*matrix_dim+j]) > 0.0001)
          {
            printf("dismatch at (%d, %d): (o)%f (n)%f\n", i, j, m[i*matrix_dim+j], tmp[i*matrix_dim+j]);
            verify = 0;
          }
      }
  }
  free(tmp);
  if (verify)
    return RET_SUCCESS;
  else
    return RET_FAILURE;
}

void
matrix_duplicate(float *src, float **dst, int matrix_dim) {
    int s = matrix_dim*matrix_dim*sizeof(float);
   float *p = (float *) malloc (s);
   memcpy(p, src, s);
   *dst = p;
}

void
print_matrix(float *m, int matrix_dim) {
    int i, j;
    for (i=0; i<matrix_dim;i++) {
      for (j=0; j<matrix_dim;j++)
        printf("%f ", m[i*matrix_dim+j]);
      printf("\n");
    }
}


// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

func_ret_t
create_matrix(float **mp, int size){
  float *m;
  int i,j;
  float lamda = -0.001;
  float *coe = (float*)alloca(sizeof(float)*2*size-1);
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }

#ifdef USE_AOT
  m = (float*) alignedMalloc(sizeof(float)*size*size);
#else
  m = (float*) malloc(sizeof(float)*size*size);
#endif
  
  if ( m == NULL) {
      return RET_FAILURE;
  }

  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }

  *mp = m;

  return RET_SUCCESS;
}
