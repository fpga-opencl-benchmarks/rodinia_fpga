#define AMB_TEMP (80.0f)

// Block size
#ifndef BSIZE
	#ifdef AMD
		#define BSIZE 16
	#else
		#define BSIZE 512
	#endif
#endif

#ifndef BLOCK_X
	#define BLOCK_X 16
#endif

#ifndef BLOCK_Y
	#define BLOCK_Y 16
#endif

// Vector size
#ifndef SSIZE
	#ifdef AOCL_BOARD_de5net_a7
		#define SSIZE 8
	#elif AOCL_BOARD_p385a_sch_ax115
		#define SSIZE 8
	#elif AMD
		#define SSIZE 4
	#endif
#endif

// Radius of stencil, e.g 5-point stencil => 1
#ifndef RAD
  #define RAD  1
#endif

// Number of parallel time steps
#ifndef TIME
	#ifdef AOCL_BOARD_de5net_a7
		#define TIME 6
	#elif AOCL_BOARD_p385a_sch_ax115
		#define TIME 21
	#elif AMD
		#define TIME 1
	#endif
#endif

// Padding to fix alignment for time steps that are not a multiple of 8
#ifndef PAD
  #define PAD TIME % 16
#endif

#define HALO_SIZE		TIME * RAD			// halo size
#define BACK_OFF		2 * HALO_SIZE			// back off for going to next block
