#define AMB_TEMP (80.0f)

#define WG_SIZE_X (64)
#define WG_SIZE_Y (4)

#ifdef AMD
	#define BSIZE 16
	#define SSIZE 1
	#define TIME 1
#endif

// Block size
#ifdef BSIZE
	#define BLOCK_X BSIZE
	#define BLOCK_Y BSIZE
#endif

#ifndef BLOCK_X
	#define BLOCK_X 128
#endif

#ifndef BLOCK_Y
	#define BLOCK_Y 128
#endif

// Vector size
#ifndef SSIZE
	#ifdef AOCL_BOARD_de5net_a7
		#define SSIZE 8
	#elif AOCL_BOARD_a10pl4_dd4gb_gx115
		#define SSIZE 8
	#elif AOCL_BOARD_p385a_sch_ax115
		#define SSIZE 8
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
	#elif AOCL_BOARD_a10pl4_dd4gb_gx115
		#define TIME 21
	#elif AOCL_BOARD_p385a_sch_ax115
		#define TIME 21
	#endif
#endif

// Padding to fix alignment for time steps that are not a multiple of 8
#ifndef PAD
  #define PAD TIME % 16
#endif

#define HALO_SIZE		TIME * RAD			// halo size
#define BACK_OFF		2 * HALO_SIZE			// back off for going to next block
