#ifndef WORK_GROUP_SIZE_H_
#define WORK_GROUP_SIZE_H_

#ifndef BSIZE
#ifdef RD_WG_SIZE_0_0
	#define BSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BSIZE RD_WG_SIZE
#else
	#define BSIZE 16
#endif 
#endif // BSIZE

#ifndef PAR
	#define PAR 4
#endif

#endif 
