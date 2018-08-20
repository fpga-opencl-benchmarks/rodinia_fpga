#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include "../../common/timer.h"
#include "../../common/power_cpu.h"

void run(int argc, char** argv);

// Timer
TimeStamp start, end;
double totalTime = 0;
double energyStart, energyEnd, totalEnergy;

//#define BENCH_PRINT

int rows, cols;
int* data;
int** wall;
int* result;
FILE *resultFile;
char* ofile = NULL;
bool write_out = 0;
#define M_SEED 9

void
init(int argc, char** argv)
{
	if(argc==3 || argc==4){
		cols = atoi(argv[1]);
		rows = atoi(argv[2]);
		if (argc==4){
			ofile = argv[3];
			write_out = 1;
		}
	}else{
                printf("Usage: %d width num_of_steps output_file\n", argv[0]);
                exit(0);
        }
	data = new int[rows*cols];
	wall = new int*[rows];
	for(int n=0; n<rows; n++)
		wall[n]=data+cols*n;
	result = new int[cols];
	
	int seed = M_SEED;
	srand(seed);

	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            wall[i][j] = rand() % 10;
        }
    }
    for (int j = 0; j < cols; j++)
        result[j] = wall[0][j];
#ifdef BENCH_PRINT
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%d ",wall[i][j]) ;
        }
        printf("\n") ;
    }
#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int main(int argc, char** argv)
{
    init(argc, argv);

    if (write_out)
    {
        resultFile = fopen(ofile, "w");
        if (resultFile == NULL)
        {
            printf("Failed to open result file!\n");
            exit(-1);
        }
    }

    unsigned long long cycles;

    int *src, *dst, *temp;
    int min;

    dst = result;
    src = new int[cols];

    energyStart = GetEnergyCPU();
    GetTime(start);
    for (int t = 0; t < rows-1; t++) {
        temp = src;
        src = dst;
        dst = temp;
        #pragma omp parallel for private(min)
        for(int n = 0; n < cols; n++){
          min = src[n];
          if (n > 0)
            min = MIN(min, src[n-1]);
          if (n < cols-1)
            min = MIN(min, src[n+1]);
          dst[n] = wall[t+1][n]+min;
        }
    }
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

    if (write_out)
    {
        #ifdef BENCH_PRINT
        for (int i = 0; i < cols; i++)
        {
            fprintf(resultFile, "%d ", data[i]);
        }
        fprintf(resultFile, "\n") ;
        #endif
    
        for (int i = 0; i < cols; i++)
        {
            fprintf(resultFile, "%d\n", dst[i]);
        }
        fclose(resultFile);
    }

    delete [] data;
    delete [] wall;
    delete [] dst;
    delete [] src;
    
    return EXIT_SUCCESS;
}
