#ifndef BENCHMARK_CU
#define BENCHMARK_CU


#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <iostream>
#include <qd/qd_real.h>
#include <qd/fpu.h>
#include <omp.h>
#include <stdexcept>
#include <cuda.h>
#include <vector>
#include <memory>
#include <iomanip>

#include "gqd.cu"
#include "cuda_util.h"
#include "cuda_util.h"
#include "test_util.h"
#include "test_common.h"
#include "gqdtest.h"

using namespace std;


#define START_MSG(msg) cout << endl << endl << msg << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;

enum OP_TYPE {UNARY = 1, BINARY = 2};


typedef gdd_real (*gdd_unary_func)(gdd_real);
typedef gdd_real (*gdd_binary_func)(const gdd_real&, const gdd_real&);


template<typename T_R, typename T_A, typename T_B>
__device__
T_R op_add(const T_A& a, const T_B& b) {
	return a + b;
}

template<typename T_R, typename T_A>
__host__ __device__
T_R op_exp(const T_A a) {
	return exp(a);
}


__device__ gdd_unary_func gdd_pexp = op_exp<gdd_real, gdd_real>;
__device__ gdd_binary_func gdd_padd = op_add<gdd_real, gdd_real, gdd_real>;


template<typename T_OUT, typename T_IN1, class T_OP>
__global__
void benchmark_kernel(T_OUT* d_out, T_IN1* d_in1, T_OP* op, const int numElement) {
	const int numTotalThread = blockDim.x*gridDim.x;
	const int threadOffset = blockIdx.x*blockDim.x + threadIdx.x;

	for(int i = threadOffset; i < numElement; i += numTotalThread) {
		d_out[i] = op[0](d_in1[i]);
	}
}


template<typename T_OUT, typename T_IN1, typename T_IN2, class T_OP>
__global__
void benchmark_kernel(T_OUT* d_out, T_IN1* d_in1, T_IN2* d_in2, T_OP* op, const int numElement) {
	const int numTotalThread = blockDim.x*gridDim.x;
	const int threadOffset = blockIdx.x*blockDim.x + threadIdx.x;

	for(int i = threadOffset; i < numElement; i += numTotalThread) {
		d_out[i] = op[0](d_in1[i], d_in2[i]);
	}
}


template<class OP>
void host_kernel(gdd_real* out, gdd_real* in1, OP op, const int numElement) {
	dd_real* t_out = new dd_real[numElement];
	dd_real* t_in1 = new dd_real[numElement];

	gqd2qd(in1, t_in1, numElement);
	for(int i = 0; i < numElement; i += 1) {
		t_out[i] = op(t_in1[i]);
	}
	qd2gqd(t_out, out, numElement);

	delete[] t_out;
	delete[] t_in1;
}



template<typename T_OUT, typename T_IN1, class T_H_OP, class T_D_OP>
void benchmark(const int numElement, T_H_OP h_op, T_D_OP &d_op) {
	cout << "#elements: " << numElement << endl;

	// Allocate host memory for operands
	T_IN1* h_in1 = new T_IN1[numElement];

	// Generate randome numbers [-1, 1] for the operands on the host
	randArray(h_in1, numElement, -1.0, 1.0);	

	// Copy the input data to the device
	T_IN1* d_in1 = NULL;
	GPUMALLOC((void**)&d_in1, sizeof(T_IN1)*numElement);
	TOGPU(d_in1, h_in1, sizeof(T_IN1)*numElement);

	// Allocate memory for results on the device 
	T_OUT* d_out = NULL;
	GPUMALLOC((void**)&d_out, sizeof(T_OUT)*numElement);

	// Assign the device function pointer
	T_D_OP* h_f = (T_D_OP*)malloc(sizeof(T_D_OP));
	T_D_OP* d_f = NULL;
	GPUMALLOC((void**)&d_f, sizeof(T_D_OP));
	checkCudaErrors(cudaMemcpyFromSymbol(h_f, d_op, sizeof(T_D_OP)));
	checkCudaErrors(cudaMemcpy(d_f, h_f, sizeof(T_D_OP), cudaMemcpyHostToDevice));

	// Performance computation on device
	CUDATimer gtimer;
	gtimer.go();
	benchmark_kernel<T_OUT, T_IN1, T_D_OP><<<512, 512>>>(d_out, d_in1, d_f, numElement);
	getLastCudaError("benchmark_kernel");
	gtimer.stop();
	cout << "GPU time: " << setprecision(2) << gtimer.report()/1000.0 << " sec" << endl;

	// Copy result back from the GPU
	T_OUT* h_out = new T_OUT[numElement];	
	FROMGPU(h_out, d_out, sizeof(T_OUT)*numElement);

	// Performance computation on host
	T_OUT* gold_out = new T_OUT[numElement];
	CPUTimer ctimer;
	ctimer.go();
	host_kernel(gold_out, h_in1, h_op, numElement);
	ctimer.stop();
	cout << "CPU time: " << setprecision(2) << ctimer.report()/1000.0 << " sec" << endl;
	
	// Check results
    checkTwoArray(h_out, gold_out, numElement);

	// Memory cleanup
	delete[] h_in1;
	delete[] h_out;
	GPUFREE(d_in1);
	GPUFREE(d_out);
}


int main(int argc, char** argv) {
    unsigned int old_cw;

	// Turn on
    fpu_fix_start(&old_cw);
	GDDStart();	


    printf("==================================================================\n");
    printf("******************** double-double precision *********************\n");
    printf("==================================================================\n");
    
	const int numElement = 10;
	START_MSG("exp");	
	benchmark<gdd_real, gdd_real>(numElement, &op_exp<dd_real, dd_real>, gdd_pexp);

	// Shutdown
    GQDEnd();
    fpu_fix_end(&old_cw);

	return EXIT_SUCCESS;
}


#endif /*BENCHMARK_CU*/
