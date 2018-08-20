#include "../problem_size.h"

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

// Doesn't seem to be used
#if 0 
// double precision support (switch between as needed for NVIDIA/AMD)
#ifdef AMDAPP
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#endif

// clBuildProgram compiler cannot link this file for some reason, so had to redefine constants and structures below
// #include ../common.h						// (in directory specified to compiler)			main function header

//======================================================================================================================================================150
//	DEFINE (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// change to double if double precision needed
//#define fp float

//#define DEFAULT_ORDER_2 256

//======================================================================================================================================================150
//	STRUCTURES (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// ???
typedef struct knode2 {
  int location;
  int indices [DEFAULT_ORDER_2 + 1];
  int  keys [DEFAULT_ORDER_2 + 1];
  bool is_leaf;
  int num_keys;
} knode2; 

//========================================================================================================================================================================================================200
//	findRangeK function
//========================================================================================================================================================================================================200

__kernel void 
findRangeK(	long height,
                __global knode2 * restrict knodesD,
                long knodes_elem,
                __global long * restrict currKnodeD,
                __global long * restrict offsetD,
                __global long * restrict lastKnodeD,
                __global long * restrict offset_2D,
                __global int * restrict startD,
                __global int * restrict endD,
                __global int * restrict RecstartD, 
                __global int * restrict ReclenD,
                int count,
                int order) {
#ifdef FINDRANGEK_UNROLL      
  int count_even = count & (~1);
#else
  int count_even = count;
#endif  
  for(int i = 0; i < height; i++){
#ifdef FINDRANGEK_UNROLL
#pragma unroll 2
#endif    
    for (int bid = 0; bid < count_even; ++bid) {
      for (int thid = 0; thid < order; thid++) {      
        if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
          // this conditional statement is inserted to avoid crush due to but in original code
          // "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
          if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){
            offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
            break;
          }
        }
      }
      // set for next tree level
      currKnodeD[bid] = offsetD[bid];

      for (int thid = 0; thid < order; thid++) {            
        if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
          // this conditional statement is inserted to avoid crush due to but in original code
          // "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
          if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){
            offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
            break;
          }
        }
      }

      lastKnodeD[bid] = offset_2D[bid];
    }
#ifdef FINDRANGEK_UNROLL
    int bid = count_even;
    if (bid < count) {
      for (int thid = 0; thid < order; thid++) {      
        if((knodesD[currKnodeD[bid]].keys[thid] <= startD[bid]) && (knodesD[currKnodeD[bid]].keys[thid+1] > startD[bid])){
          // this conditional statement is inserted to avoid crush due to but in original code
          // "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
          if(knodesD[currKnodeD[bid]].indices[thid] < knodes_elem){
            offsetD[bid] = knodesD[currKnodeD[bid]].indices[thid];
            break;
          }
        }
      }
      // set for next tree level
      currKnodeD[bid] = offsetD[bid];

      for (int thid = 0; thid < order; thid++) {            
        if((knodesD[lastKnodeD[bid]].keys[thid] <= endD[bid]) && (knodesD[lastKnodeD[bid]].keys[thid+1] > endD[bid])){
          // this conditional statement is inserted to avoid crush due to but in original code
          // "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
          // more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
          if(knodesD[lastKnodeD[bid]].indices[thid] < knodes_elem){
            offset_2D[bid] = knodesD[lastKnodeD[bid]].indices[thid];
            break;
          }
        }
      }

      lastKnodeD[bid] = offset_2D[bid];
    }
#endif // FINDRANGEK_UNROLL    
  }

  for (int bid = 0; bid < count; ++bid) {  
    // Find the index of the starting record
    for (int thid = 0; thid < order; thid++) {
      if(knodesD[currKnodeD[bid]].keys[thid] == startD[bid]){
        RecstartD[bid] = knodesD[currKnodeD[bid]].indices[thid];
        break;
      }
    }

    // Find the index of the ending record
    for (int thid = 0; thid < order; thid++) {    
      if(knodesD[lastKnodeD[bid]].keys[thid] == endD[bid]){
        ReclenD[bid] = knodesD[lastKnodeD[bid]].indices[thid] - RecstartD[bid]+1;
        break;
      }
    }
  }

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
