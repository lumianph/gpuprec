/* 
 * File:   cuda_util.h
 * Author: Mian
 *
 * Created on April 12, 2012, 9:07 AM
 */

#ifndef MIAN_CUDA_UTIL_H
#define	MIAN_CUDA_UTIL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <stdint.h>

#include "timer.h"


/*constants*/
#define NUM_TOTAL_THREAD	(gridDim.x*blockDim.x)
#define GLOBAL_THREAD_OFFSET	(blockDim.x*blockIdx.x + threadIdx.x)

namespace CUDAUtil {


    /*timing functions*/
    //return ms, rather than second!

    inline void startTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
    }

    inline float endTimer(cudaEvent_t& start, cudaEvent_t& stop) {
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime = 0;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        return elapsedTime;
    }


    class CUDATimer : public Timer {
    private:
        cudaEvent_t _start, _stop;
    public:

        CUDATimer() : _start(), _stop(){
        };

        inline void go() {
            startTimer(_start, _stop);
        };

        inline void stop() {
            _t += endTimer(_start, _stop);
        };
    };



    /*CUDA helper functions*/

    // This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

    inline void __checkCudaErrors(cudaError err, const char *file, const int line) {
        if (cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                    file, line, (int) err, cudaGetErrorString(err));
            exit(-1);
        }
    }

    // This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

    inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                    file, line, errorMessage, (int) err, cudaGetErrorString(err));
            exit(-1);
        }
    }

    template<class T>
    inline void mallocAndCopyToDevice(T* &d_dest, const T* h_src, const size_t size) {
        checkCudaErrors(cudaMalloc(&d_dest, size));
        checkCudaErrors(cudaMemcpy(d_dest, h_src, size, cudaMemcpyHostToDevice));
    };
    
    template<class T>
    inline void mallocAndCopyPointersToDevice(T** &d_ptr, const T* d_mem, 
            const T* const* h_ptr, const T* h_mem, const size_t numPtr) {
        T** h_tempPtr = (T**)malloc(sizeof(T*)*numPtr);
        for(size_t i = 0; i < numPtr; i += 1) {
            h_tempPtr[i] = const_cast<T*>(d_mem) + (h_ptr[i] - h_mem);
        }
        mallocAndCopyToDevice(d_ptr, h_tempPtr, sizeof(T*)*numPtr);
        
        delete[] h_tempPtr;
    }

#define CUDA_FREE(ptr) checkCudaErrors(cudaFree(ptr))
    
} /*namespace CUDAUtil*/

#endif	/* MIAN_CUDA_UTIL_H */

