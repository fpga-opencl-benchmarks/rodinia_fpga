#ifndef PROBLEM_SIZE_H_
#define PROBLEM_SIZE_H_

#ifdef RD_WG_SIZE_0_0
        #define  DEFAULT_ORDER RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define  DEFAULT_ORDER RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define  DEFAULT_ORDER RD_WG_SIZE
#else
        #define  DEFAULT_ORDER 256
#endif

#ifdef RD_WG_SIZE_1_0
        #define  DEFAULT_ORDER_2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define  DEFAULT_ORDER_2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define  DEFAULT_ORDER_2 RD_WG_SIZE
#else
        #define  DEFAULT_ORDER_2 256
#endif

#endif // PROBLEM_SIZE_H_
