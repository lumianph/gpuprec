
#ifndef _CUDA_HEADER_CU_
#define _CUDA_HEADER_CU_

#include <stdio.h>
#include <stdlib.h>
#include <cutil_inline.h>


/** for CUDA 2.0 */
#ifdef CUDA_2
    #define cutilCheckMsg CUT_CHECK_ERROR
    #define cutilSafeCall CUDA_SAFE_CALL
#endif


/* kernel macros */
#define NUM_TOTAL_THREAD (gridDim.x*blockDim.x)
#define GLOBAL_THREAD_OFFSET (blockDim.x*blockIdx.x + threadIdx.x)

/** macro utility */
#define GPUMALLOC(D_DATA, MEM_SIZE) cutilSafeCall(cudaMalloc(D_DATA, MEM_SIZE))
#define TOGPU(D_DATA, H_DATA, MEM_SIZE) cutilSafeCall(cudaMemcpy(D_DATA, H_DATA, MEM_SIZE, cudaMemcpyHostToDevice))
#define FROMGPU( H_DATA, D_DATA, MEM_SIZE ) cutilSafeCall(cudaMemcpy( H_DATA, D_DATA, MEM_SIZE, cudaMemcpyDeviceToHost))
#define GPUTOGPU( DEST, SRC, MEM_SIZE ) cutilSafeCall(cudaMemcpy( DEST, SRC, MEM_SIZE, cudaMemcpyDeviceToDevice ))
#define GPUFREE( MEM ) cutilSafeCall(cudaFree(MEM));


/** timer utility */
void startTimer( unsigned int*  timer ) {
    	*timer = 0;
	CUT_SAFE_CALL( cutCreateTimer( timer));
	CUT_SAFE_CALL( cutStartTimer( *timer));
}

float endTimer( unsigned int*  timer, char* title ) {
    	CUT_SAFE_CALL( cutStopTimer( *timer));
	float ms = cutGetTimerValue(*timer);	
	printf( "*** %s processing time: %.3f sec ***\n", title, ms/1000.0);
	CUT_SAFE_CALL( cutDeleteTimer( *timer));

	return ms;
}


#endif

