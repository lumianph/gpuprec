#ifndef BENCHMARK_KERNEL_CU
#define BENCHMARK_KERNEL_CU


#include "gqd.cu"
#include "cuda_util.h"

template<typename T_OUT, typename T_IN1, class T_OP>
__global__
void kernel(T_OUT d_out, T_IN1 d_in1, T_OP op, const int numElement) {
	const int numTotalThread = blockDim.x*gridDim.x;
	const int threadOffset = blockIdx.x*blockDim.x + threadIdx.x;

	for(int i = threadOffset; i < numElement; i += numTotalThread) {
		d_out[i] = op(d_in1[i]);
	}
}

template<typename T_OUT, typename T_IN1, class T_OP>
void benchmark_kernel(T_OUT d_out, T_IN1 d_in1, T_OP op, const int numElement) {
	kernel<T_OUT, T_IN1, T_OP><<<512, 512>>>(d_out, d_in1, op, numElement);
	getLastCudaError("kernel");	
}

#endif /*BENCHMARK_KERNEL_CU*/
