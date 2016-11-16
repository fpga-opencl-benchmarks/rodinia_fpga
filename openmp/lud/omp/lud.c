/*
 * =====================================================================================
 *
 *       Filename:  suite.c
 *
 *    Description:  The main wrapper for the suite
 *
 *        Version:  1.0
 *        Created:  10/22/2009 08:40:34 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Liang Wang (lw2aw), lw2aw@virginia.edu
 *        Company:  CS@UVa
 *
 * =====================================================================================
 */

#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <stdlib.h>
#include <assert.h>

#include "common.h"
#include "../../../common/timer.h"
#include "../../../common/power_cpu.h"

static int do_verify = 0;
int omp_num_threads = 40;

static struct option long_options[] = {
  /* name, has_arg, flag, val */
  {"input", 1, NULL, 'i'},
  {"size", 1, NULL, 's'},
  {"verify", 0, NULL, 'v'},
  {0,0,0,0}
};

extern void
lud_omp(float *m, int matrix_dim);

int
main ( int argc, char *argv[] )
{
  int matrix_dim = 32; /* default size */
  int opt, option_index=0;
  func_ret_t ret;
  const char *input_file = NULL;
  float *m, *mm;
  TimeStamp start, end;
  double totalTime = 0;
  double energyStart, energyEnd, totalEnergy;

  while ((opt = getopt_long(argc, argv, "::vs:n:i:", 
                            long_options, &option_index)) != -1 ) {
    switch(opt){
    case 'i':
      input_file = optarg;
      break;
    case 'v':
      do_verify = 1;
      break;
    case 'n':
      omp_num_threads = atoi(optarg);
      break;
    case 's':
      matrix_dim = atoi(optarg);
      printf("Generate input matrix internally, size =%d\n", matrix_dim);
      // fprintf(stderr, "Currently not supported, use -i instead\n");
      // fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      // exit(EXIT_FAILURE);
      break;
    case '?':
      fprintf(stderr, "invalid option\n");
      break;
    case ':':
      fprintf(stderr, "missing argument\n");
      break;
    default:
      fprintf(stderr, "Usage: %s [-v] [-s matrix_size|-i input_file]\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }
  
  if ( (optind < argc) || (optind == 1)) {
    fprintf(stderr, "Usage: %s [-v] [-n no. of threads] [-s matrix_size|-i input_file]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  if (input_file) {
    printf("Reading matrix from file %s\n", input_file);
    ret = create_matrix_from_file(&m, input_file, &matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix from file %s\n", input_file);
      exit(EXIT_FAILURE);
    }
  }
  else if (matrix_dim) {
    printf("Creating matrix internally size=%d\n", matrix_dim);
    ret = create_matrix(&m, matrix_dim);
    if (ret != RET_SUCCESS) {
      m = NULL;
      fprintf(stderr, "error create matrix internally size=%d\n", matrix_dim);
      exit(EXIT_FAILURE);
    }
  }
 
  else {
    printf("No input file specified!\n");
    exit(EXIT_FAILURE);
  } 

  if (do_verify){
    //printf("Before LUD\n");
    /* print_matrix(m, matrix_dim); */
    matrix_duplicate(m, &mm, matrix_dim);
  }


  energyStart = GetEnergyCPU();
  GetTime(start);
  
  lud_omp(m, matrix_dim);
  
  GetTime(end);
  energyEnd = GetEnergyCPU();
  totalTime = TimeDiff(start, end);
  totalEnergy = energyEnd - energyStart;
  
  printf("Computation done in %0.3lf ms.\n", totalTime);
  if (energyStart != -1) // -1 --> failed to read energy values
  {
    printf("Total energy used is %0.3lf jouls.\n", totalEnergy);
    printf("Average power consumption is %0.3lf watts.\n", totalEnergy/(totalTime/1000.0));
  }

  if (do_verify){
    //printf("After LUD\n");
    /* print_matrix(m, matrix_dim); */
    printf("Verifying output: ");
    if (lud_verify(mm, m, matrix_dim) == RET_SUCCESS)
    {
      printf("verification succeeded!\n");
    }
    else
    {
      printf("verification failed!\n");
    }
    free(mm);
  }
  
  free(m);

  return EXIT_SUCCESS;
}				/* ----------  end of function main  ---------- */
