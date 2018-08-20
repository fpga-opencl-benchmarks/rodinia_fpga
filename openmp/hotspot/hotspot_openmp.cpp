#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//using namespace std;

#define BLOCK_SIZE 16
#define BLOCK_SIZE_C BLOCK_SIZE
#define BLOCK_SIZE_R BLOCK_SIZE

#define STR_SIZE	256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
//#define OPEN
//#define NUM_THREAD 4

typedef float FLOAT;

#include "../../common/timer.h"
#include "../../common/power_cpu.h"

TimeStamp start, end;
double totalTime = 0;
double energyStart, energyEnd, totalEnergy;

/* chip parameters	*/
const FLOAT t_chip = 0.0005;
const FLOAT chip_height = 1.6;
const FLOAT chip_width = 1.6;

#ifdef OMP_OFFLOAD
	#pragma offload_attribute(push, target(mic))
#endif

/* ambient temperature, assuming no package at all	*/
const FLOAT amb_temp = 80.0;

int num_omp_threads;

/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void single_iteration(FLOAT *result, FLOAT *temp, FLOAT *power, int row, int col,
					  FLOAT Cap_1, FLOAT Rx_1, FLOAT Ry_1, FLOAT Rz_1, 
					  FLOAT step)
{
    FLOAT delta = 0;
    int r, c;
    int chunk;
    int num_chunk = row*col / (BLOCK_SIZE_R * BLOCK_SIZE_C);
    int chunks_in_row = col/BLOCK_SIZE_C;
    int chunks_in_col = row/BLOCK_SIZE_R;

#ifdef OPEN
    #ifndef __MIC__
	omp_set_num_threads(num_omp_threads);
    #endif
    #pragma omp parallel for shared(power, temp, result) private(chunk, r, c, delta) firstprivate(row, col, num_chunk, chunks_in_row) schedule(static)
#endif
    for ( chunk = 0; chunk < num_chunk; ++chunk )
    {
        int r_start = BLOCK_SIZE_R*(chunk/chunks_in_col);
        int c_start = BLOCK_SIZE_C*(chunk%chunks_in_row); 
        int r_end = r_start + BLOCK_SIZE_R > row ? row : r_start + BLOCK_SIZE_R;
        int c_end = c_start + BLOCK_SIZE_C > col ? col : c_start + BLOCK_SIZE_C;
       
        if ( r_start == 0 || c_start == 0 || r_end == row || c_end == col )
        {
            for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
                for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
                    /* Corner 1 */
                    if ( (r == 0) && (c == 0) ) {
                        delta = (Cap_1) * (power[0] +
                            (temp[1] - temp[0]) * Rx_1 +
                            (temp[col] - temp[0]) * Ry_1 +
                            (amb_temp - temp[0]) * Rz_1);
                    }	/* Corner 2 */
                    else if ((r == 0) && (c == col-1)) {
                        delta = (Cap_1) * (power[c] +
                            (temp[c-1] - temp[c]) * Rx_1 +
                            (temp[c+col] - temp[c]) * Ry_1 +
                        (   amb_temp - temp[c]) * Rz_1);
                    }	/* Corner 3 */
                    else if ((r == row-1) && (c == col-1)) {
                        delta = (Cap_1) * (power[r*col+c] + 
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
                        (   amb_temp - temp[r*col+c]) * Rz_1);					
                    }	/* Corner 4	*/
                    else if ((r == row-1) && (c == 0)) {
                        delta = (Cap_1) * (power[r*col] + 
                            (temp[r*col+1] - temp[r*col]) * Rx_1 + 
                            (temp[(r-1)*col] - temp[r*col]) * Ry_1 + 
                            (amb_temp - temp[r*col]) * Rz_1);
                    }	/* Edge 1 */
                    else if (r == 0) {
                        delta = (Cap_1) * (power[c] + 
                            (temp[c+1] + temp[c-1] - 2.0*temp[c]) * Rx_1 + 
                            (temp[col+c] - temp[c]) * Ry_1 + 
                            (amb_temp - temp[c]) * Rz_1);
                    }	/* Edge 2 */
                    else if (c == col-1) {
                        delta = (Cap_1) * (power[r*col+c] + 
                            (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.0*temp[r*col+c]) * Ry_1 + 
                            (temp[r*col+c-1] - temp[r*col+c]) * Rx_1 + 
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }	/* Edge 3 */
                    else if (r == row-1) {
                        delta = (Cap_1) * (power[r*col+c] + 
                            (temp[r*col+c+1] + temp[r*col+c-1] - 2.0*temp[r*col+c]) * Rx_1 + 
                            (temp[(r-1)*col+c] - temp[r*col+c]) * Ry_1 + 
                            (amb_temp - temp[r*col+c]) * Rz_1);
                    }	/* Edge 4 */
                    else if (c == 0) {
                        delta = (Cap_1) * (power[r*col] + 
                            (temp[(r+1)*col] + temp[(r-1)*col] - 2.0*temp[r*col]) * Ry_1 + 
                            (temp[r*col+1] - temp[r*col]) * Rx_1 + 
                            (amb_temp - temp[r*col]) * Rz_1);
                    }
                    result[r*col+c] =temp[r*col+c]+ delta;
                }
            }
            continue;
        }

        for ( r = r_start; r < r_start + BLOCK_SIZE_R; ++r ) {
#pragma omp simd        
            for ( c = c_start; c < c_start + BLOCK_SIZE_C; ++c ) {
            /* Update Temperatures */
                result[r*col+c] =temp[r*col+c]+ 
                     ( Cap_1 * (power[r*col+c] + 
                    (temp[(r+1)*col+c] + temp[(r-1)*col+c] - 2.f*temp[r*col+c]) * Ry_1 + 
                    (temp[r*col+c+1] + temp[r*col+c-1] - 2.f*temp[r*col+c]) * Rx_1 + 
                    (amb_temp - temp[r*col+c]) * Rz_1));
            }
        }
    }
}

#ifdef OMP_OFFLOAD
	#pragma offload_attribute(pop)
#endif

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(FLOAT *result, int num_iterations, FLOAT *temp, FLOAT *power, int row, int col) 
{
	#ifdef VERBOSE
	int i = 0;
	#endif

	FLOAT grid_height = chip_height / row;
	FLOAT grid_width = chip_width / col;

	FLOAT Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
	FLOAT Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
	FLOAT Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
	FLOAT Rz = t_chip / (K_SI * grid_height * grid_width);

	FLOAT max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
	FLOAT step = PRECISION / max_slope / 1000.0;

	FLOAT Rx_1=1.f/Rx;
	FLOAT Ry_1=1.f/Ry;
	FLOAT Rz_1=1.f/Rz;
	FLOAT Cap_1 = step/Cap;
	#ifdef VERBOSE
	fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
	fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
	#endif

#ifdef OMP_OFFLOAD
        int array_size = row*col;
#pragma omp target \
        map(temp[0:array_size]) \
        map(to: power[0:array_size], row, col, Cap_1, Rx_1, Ry_1, Rz_1, step, num_iterations) \
        map( result[0:array_size])
#endif
	{
		FLOAT* r = result;
		FLOAT* t = temp;
		energyStart = GetEnergyCPU();
		GetTime(start);
		for (int i = 0; i < num_iterations ; i++)
		{
			#ifdef VERBOSE
			fprintf(stdout, "iteration %d\n", i++);
			#endif
			single_iteration(r, t, power, row, col, Cap_1, Rx_1, Ry_1, Rz_1, step);
			FLOAT* tmp = t;
			t = r;
			r = tmp;
		}
		GetTime(end);
		energyEnd = GetEnergyCPU();
		totalTime = TimeDiff(start, end);
		totalEnergy = energyEnd - energyStart;
        }
	#ifdef VERBOSE
	fprintf(stdout, "iteration %d\n", i++);
	#endif
}

void fatal(char const* s)
{
	fprintf(stderr, "error: %s\n", s);
	exit(1);
}

void read_input(FLOAT *vect, int grid_rows, int grid_cols, char *file)
{
  	int i;
	FILE *fp;
	char str[STR_SIZE];
	FLOAT val;

	fp = fopen (file, "r");
	if (!fp)
		fatal ("file could not be opened for reading");

	for (i=0; i < grid_rows * grid_cols; i++) {
		fgets(str, STR_SIZE, fp);
		if (feof(fp))
			fatal("not enough lines in file");
		if ((sscanf(str, "%f", &val) != 1) )
			fatal("invalid file format");
		vect[i] = val;
	}

	fclose(fp);	
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n", argv[0]);
	fprintf(stderr, "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
	fprintf(stderr, "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
	fprintf(stderr, "\t<sim_time>   - number of iterations\n");
	fprintf(stderr, "\t<no. of threads>   - number of threads\n");
	fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
	fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
        fprintf(stderr, "\t<output_file> - name of the output file\n");
	exit(1);
}

int main(int argc, char **argv)
{
	int grid_rows, grid_cols, sim_time, i;
	int write_out = 0;
	FLOAT *temp, *power, *result;
	char *tfile, *pfile, *ofile;
	
	/* check validity of inputs	*/
	if (argc != 8 && argc != 7)
		usage(argc, argv);
	if ((grid_rows = atoi(argv[1])) <= 0 ||
		(grid_cols = atoi(argv[2])) <= 0 ||
		(sim_time = atoi(argv[3])) <= 0 || 
		(num_omp_threads = atoi(argv[4])) <= 0
		)
		usage(argc, argv);

	/* allocate memory for the temperature and power arrays	*/
	temp = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	power = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	result = (FLOAT *) calloc (grid_rows * grid_cols, sizeof(FLOAT));
	if(!temp || !power)
		fatal("unable to allocate memory");

	/* read initial temperatures and input power	*/
	tfile = argv[5];
	pfile = argv[6];
	if (argc == 8)
	{
		write_out = 1;
		ofile=argv[7];
	}

	read_input(temp, grid_rows, grid_cols, tfile);
	read_input(power, grid_rows, grid_cols, pfile);

	printf("Start computing the transient temperature\n");
	
	compute_tran_temp(result,sim_time, temp, power, grid_rows, grid_cols);

	printf("Ending simulation\n\n");
	printf("Computation done in %0.3lf ms.\n", totalTime);
	if (energyStart != -1) // -1 --> failed to read energy values
	{
		printf("Total energy used is %0.3lf jouls.\n", totalEnergy);
		printf("Average power consumption is %0.3lf watts.\n", totalEnergy/(totalTime/1000.0));
	}

	/* output results	*/
#ifdef VERBOSE
	fprintf(stdout, "Final Temperatures:\n");
#endif

	if (write_out)
	{
		FILE *ofile_fp = fopen(ofile, "w");
		if (!ofile_fp)
		{
			printf("could not open output file, skipping...\n");
		}
		else
		{
			for(i=0; i < grid_rows * grid_cols; i++)
			{
				fprintf(ofile_fp, "%d\t%f\n", i, temp[i]);
			}
			fclose(ofile_fp);
		}
	}

	/* cleanup	*/
	free(temp);
	free(power);

	return 0;
}
