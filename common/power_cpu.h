// Based on earlier work by Vince Weaver (vincent*dot*weaver*at*maine*dot*edu) and
// Romain Dolbeau (romain*at*dolbeau*dot*org). Original code available at
// http://web.eece.maine.edu/~vweaver/projects/rapl/rapl-read.c
// Root access is needed to access /dev/cpu/??/msr driver
// Only supported on Intel Sandy-bridge and above

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <inttypes.h>
#include <unistd.h>
#include <math.h>
#include <string.h>

//====================================================================================================================================
// CPU Energy Calculator
//====================================================================================================================================

#define MSR_RAPL_POWER_UNIT	0x606
#define MSR_PKG_ENERGY_STATUS	0x611

#define CPU_SANDYBRIDGE		42
#define CPU_SANDYBRIDGE_EP	45
#define CPU_IVYBRIDGE		58
#define CPU_IVYBRIDGE_EP	62
#define CPU_HASWELL		60
#define CPU_HASWELL_EP		63
#define CPU_BROADWELL		61

static inline int open_msr(int core)
{
	char msr_filename[BUFSIZ];
	int fd;

	sprintf(msr_filename, "/dev/cpu/%d/msr", core);
	fd = open(msr_filename, O_RDONLY);
	if (fd < 0)
	{
		if (errno == ENXIO)
		{
			fprintf(stderr, "Energy calculation skipped due to: invalid CPU/Core.\n");
			return -1;
		}
		else if (errno == EIO)
		{
			fprintf(stderr, "Energy calculation skipped due to: CPU doesn't support MSR.\n");
			return -1;
		}
		else
		{
			fprintf(stderr, "Energy calculation skipped due to: failed to open MSR driver.\n");
			return -1;
		}
	}

	return fd;
}

static inline long long read_msr(int fd, int which)
{
	uint64_t data;

	if (pread(fd, &data, sizeof data, which) != sizeof data)
	{
		fprintf(stderr, "Energy calculation skipped due to: failed to read from MSR driver.\n");
		return -1;
	}

	return (long long)data;
}

static inline int detect_cpu(void)
{
	FILE *fff;

	int family, model = -1;
	char buffer[BUFSIZ], *result;
	char vendor[BUFSIZ];

	fff = fopen("/proc/cpuinfo","r");
	if (fff == NULL)
	{
		fprintf(stderr, "Energy calculation skipped due to: failed to open cpu info\n");
		return -1;
	}

	while(1)
	{
		result = fgets(buffer, BUFSIZ, fff);
		if (result == NULL)
		{
			break;
		}

		if (!strncmp(result, "vendor_id", 8))
		{
			sscanf(result, "%*s%*s%s", vendor);

			if (strncmp(vendor, "GenuineIntel", 12))
			{
				printf("Energy calculation skipped due to: not an Intel chip.\n");
				return -1;
			}
		}

		if (!strncmp(result, "cpu family", 10))
		{
			sscanf(result,"%*s%*s%*s%d", &family);
			if (family != 6) {
				printf("Energy calculation skipped due to: unsupported CPU family.\n");
				return -1;
			}
		}

		if (!strncmp(result, "model", 5))
		{
			sscanf(result, "%*s%*s%d", &model);
		}

	}

	fclose(fff);

	if (model != CPU_SANDYBRIDGE && model != CPU_SANDYBRIDGE_EP &&
	    model != CPU_IVYBRIDGE   && model != CPU_IVYBRIDGE_EP   &&
	    model != CPU_HASWELL     && model != CPU_HASWELL_EP     &&
	    model != CPU_BROADWELL)
	{
		fprintf(stderr, "Energy calculation skipped due to: unsupported CPU model.\n");
		model = -1;
	}

	return model;
}

// Returns amount of power used in joules
static inline double GetEnergyCPU()
{
	if(geteuid() != 0)
	{
		fprintf(stderr, "Energy calculation skipped due to: non-root user.\n");
		return -1;
	}
	
	int fd, core = 0;
	long long result;
	double cpu_energy_units;
	double power;
	int cpu_model;

	cpu_model = detect_cpu();
	if (cpu_model < 0)
	{
		return -1;
	}

	fd = open_msr(core);
	if (fd < 0)
	{
		return -1;
	}
	
	result = read_msr(fd,MSR_RAPL_POWER_UNIT);
	if (result < 0)
	{
		return -1;
	}
	cpu_energy_units = pow( 0.5, (double)((result >> 8) & 0x1f) );

	result = read_msr(fd, MSR_PKG_ENERGY_STATUS);
	if (result < 0)
	{
		return -1;
	}
	power = (double)result * cpu_energy_units;

	close(fd);

	return power;
}
