#ifndef __GDD_TEST_CU__
#define __GDD_TEST_CU__

#include "test_common.h"
#include "gqd.cu"

#define TEST_SINCOS


template<class T>
__global__
void device_basic_kernel(const T* d_in1, const T* d_in2, 
			 const unsigned int numElement,
                         T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = BASIC_FUNC(d_in1[i], d_in2[i]); 
        }
}


template<class T>
__global__
void device_add_kernel(const T* d_in1, const T* d_in2,
                         const unsigned int numElement,
                         T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = d_in1[i] + d_in2[i];
        }
}


template<class T>
__global__
void device_mul_kernel(const T* d_in1, const T* d_in2,
                         const unsigned int numElement,
                         T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = d_in1[i] * d_in2[i];
        }
}


template<class T>
__global__
void device_div_kernel(const T* d_in1, const T* d_in2,
                         const unsigned int numElement,
                         T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = d_in1[i] / d_in2[i];
        }
}


template<class T>
__global__
void device_sqrt_kernel(const T* d_in, const unsigned int numElement,
                          T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = sqrt(d_in[i]);
        }
}


template<class T>
__global__
void device_sqr_kernel(const T* d_in, const unsigned int numElement,
                          T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = sqr(d_in[i]);
        }
}


template<class T>
__global__
void device_exp_kernel(const T* d_in, const unsigned int numElement, 
			  T* d_out) {
	const unsigned numTotalThread = NUM_TOTAL_THREAD;
	const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;
	
	for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
		d_out[i] = exp(d_in[i]);
	}
}


template<class T>
__global__
void device_log_kernel(const T* d_in, const unsigned int numElement,
                          T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = log(d_in[i]);
        }
}


template<class T>
__global__
void device_sin_kernel(const T* d_in, const unsigned int numElement,
                          T* d_out) {

        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = sin(d_in[i]);
        }
}


template<class T>
__global__
void device_cos_kernel(const T* d_in, const unsigned int numElement,
                          T* d_out) {
	const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = cos(d_in[i]);
        }
}


template<class T>
__global__
void device_tan_kernel(const T* d_in, const unsigned int numElement,
                          T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = tan(d_in[i]);
        }
}



template<class T>
__global__
void device_atan_kernel(const T* d_in, const unsigned int numElement,
                          T* d_out) {
/*
#ifndef TEST_SINCOS
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                d_out[i] = atan(d_in[i]);
        }
#endif
*/
}



template<class T>
float device_basic_template(T* h_in1, T* h_in2, T* h_out, const unsigned int numElement,
			OPERATOR op = ADD, 
                        const unsigned int numBlock = 128, 
                        const unsigned int numThread = 128) {

        T* d_in1 = NULL;
	T* d_in2 = NULL;
        T* d_out = NULL;
        GPUMALLOC((void**)&d_in1, sizeof(T)*numElement);
	GPUMALLOC((void**)&d_in2, sizeof(T)*numElement);
        GPUMALLOC((void**)&d_out, sizeof(T)*numElement);
        TOGPU(d_in1, h_in1, sizeof(T)*numElement);
	TOGPU(d_in2, h_in2, sizeof(T)*numElement);

        unsigned int timer = 0;
	float elapsedTime = 0.0;

	if(op == ADD) {
	        startTimer(&timer);
	        device_add_kernel<<<numBlock, numThread>>>(d_in1, d_in2, numElement, d_out);
	        cutilCheckMsg("device_add_kernel");
	        cutilSafeCall(cudaThreadSynchronize());
	        elapsedTime = endTimer(&timer, "device_add_kernel");
	} else if (op == MUL) {
                startTimer(&timer);
                device_mul_kernel<<<numBlock, numThread>>>(d_in1, d_in2, numElement, d_out);
                cutilCheckMsg("device_mul_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_mul_kernel");
	} else if(op == DIV) {
                startTimer(&timer);
                device_div_kernel<<<numBlock, numThread>>>(d_in1, d_in2, numElement, d_out);
                cutilCheckMsg("device_div_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_div_kernel");
	} else {
		printf("!!!Never here!\n");
		exit(EXIT_FAILURE);
	}

        FROMGPU(h_out, d_out, sizeof(T)*numElement);

        GPUFREE(d_in1);
	GPUFREE(d_in2);
        GPUFREE(d_out);

	return elapsedTime;
}

float device_basic(gdd_real* h_in1, gdd_real* h_in2, gdd_real* h_out, const unsigned int numElement,
                        OPERATOR op = ADD,
                        const unsigned int numBlock = 128,
                        const unsigned int numThread = 128) {
	return device_basic_template(h_in1, h_in2, h_out, numElement, op, numBlock, numThread);
}


float device_basic(gqd_real* h_in1, gqd_real* h_in2, gqd_real* h_out, const unsigned int numElement,
                        OPERATOR op = ADD,
                        const unsigned int numBlock = 128,
                        const unsigned int numThread = 128) {
        return device_basic_template(h_in1, h_in2, h_out, numElement, op, numBlock, numThread);
}


template<class T>
float device_math_template(T* h_in, const unsigned int numElement, T* h_out,
		   MATH math, const unsigned int numBlock, const unsigned int numThread) {

        T* d_in = NULL;
        T* d_out = NULL;
        GPUMALLOC((void**)&d_in, sizeof(T)*numElement);
        GPUMALLOC((void**)&d_out, sizeof(T)*numElement);
        TOGPU(d_in, h_in, sizeof(T)*numElement);
        unsigned int timer = 0;
	float elapsedTime = 0.0;

	if(math == SQRT) {
        	startTimer(&timer);
	        device_sqrt_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
	        cutilCheckMsg("device_sqrt_kernel");
	        cutilSafeCall(cudaThreadSynchronize());
	        elapsedTime = endTimer(&timer, "device_sqrt_kernel");
	} else if(math == SQR) {
                startTimer(&timer);
                device_sqr_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
                cutilCheckMsg("device_sqr_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_sqr_kernel");
	} else if(math == EXP) {
                startTimer(&timer);
                device_exp_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
                cutilCheckMsg("device_exp_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_exp_kernel");
	} else if(math == LOG) {
                startTimer(&timer);
                device_log_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
                cutilCheckMsg("device_log_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_log_kernel");
	} 
	else if(math == SIN) {
                startTimer(&timer);
                device_sin_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
                cutilCheckMsg("device_sin_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_sin_kernel");
	} 
	else if(math == COS) {
                startTimer(&timer);
                device_cos_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
                cutilCheckMsg("device_cos_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_cos_kernel");
	} else if(math == TAN) {
                startTimer(&timer);
                device_tan_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
                cutilCheckMsg("device_tan_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_tan_kernel");
	} else if(math == ATAN) {
                /*startTimer(&timer);
                device_atan_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
                cutilCheckMsg("device_atan_kernel");
                cutilSafeCall(cudaThreadSynchronize());
                elapsedTime = endTimer(&timer, "device_atan_kernel");*/
        } 
	else {
		printf("!!! Never here.\n");
		exit(EXIT_FAILURE);
	}

        FROMGPU(h_out, d_out, sizeof(T)*numElement);

        GPUFREE(d_in);
        GPUFREE(d_out);

	return elapsedTime;
}


float device_math(gdd_real* h_in, const unsigned int numElement, gdd_real* h_out,
                   MATH math, const unsigned int numBlock, const unsigned int numThread) {
	return device_math_template(h_in, numElement, h_out, math, numBlock, numThread);
}


float device_math(gqd_real* h_in, const unsigned int numElement, gqd_real* h_out,
                   MATH math, const unsigned int numBlock, const unsigned int numThread) {
	return device_math_template(h_in, numElement, h_out, math, numBlock, numThread);
}


template<class T>
__global__
void device_defined_kernel(const T* d_in, const unsigned int numElement, T* d_out) {
        const unsigned numTotalThread = NUM_TOTAL_THREAD;
        const unsigned globalThreadOffset = GLOBAL_THREAD_OFFSET;

        for(unsigned int i = globalThreadOffset; i < numElement; i += numTotalThread) {
                //d_out[i] = (sin(d_in[i]));
        }
}


template<class T>
float device_defined_template(T* h_in, const unsigned int numElement, T* h_out,
                   	      const unsigned int numBlock, const unsigned int numThread) {

        T* d_in = NULL;
        T* d_out = NULL;
        GPUMALLOC((void**)&d_in, sizeof(T)*numElement);
        GPUMALLOC((void**)&d_out, sizeof(T)*numElement);
        TOGPU(d_in, h_in, sizeof(T)*numElement);
        unsigned int timer = 0;
        float elapsedTime = 0.0;

        startTimer(&timer);
        device_defined_kernel<<<numBlock, numThread>>>(d_in, numElement, d_out);
        cutilCheckMsg("device_defined_kernel");
        cutilSafeCall(cudaThreadSynchronize());
        elapsedTime = endTimer(&timer, "device_defined_kernel");

        FROMGPU(h_out, d_out, sizeof(T)*numElement);

        GPUFREE(d_in);
        GPUFREE(d_out);

        return elapsedTime;
}


float device_defined(gdd_real* h_in, const unsigned int numElement, gdd_real* h_out,
                        const unsigned int numBlock, const unsigned int numThread) {
	return device_defined_template(h_in, numElement, h_out, numBlock, numThread);
}

float device_defined(gqd_real* h_in, const unsigned int numElement, gqd_real* h_out,
                        const unsigned int numBlock, const unsigned int numThread) {
	return device_defined_template(h_in, numElement, h_out, numBlock, numThread);
}


/* the QRS map kernel */
__global__
void gpu_fx_map_kernel1( const gqd_real* d_x, gqd_real* d_c, const unsigned int N ) { 
	const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int delta = blockDim.x*gridDim.x;
	gqd_real x = d_x[0];
	gqd_real c;
	
	for( unsigned int i = index; i < N; i += delta ) {
		c = sqrt(1.0 - sqr( ((N - x)/N) * (1.0 - 2.0*(i)/(N - 1.0)) ));
		d_c[i] = 2.0*c - 1.0/c;
	}
}

void device_qrsmap(const unsigned int N, const int numBlock, const int numThread) {
	printf("numBlock = %d, numThread = %d\n", numBlock, numThread);

	gqd_real h_x;
	h_x.x = 0.6; 
	h_x.y = 0.0; 
	h_x.z = 0.0; 
	h_x.w = 0.0;
	
	gqd_real* d_x = NULL;
	GPUMALLOC((void**)&d_x, sizeof(gqd_real));
	TOGPU( d_x, &h_x, sizeof(gqd_real) );
	gqd_real* d_c = NULL;
	GPUMALLOC((void**)&d_c, sizeof(gqd_real)*N);
	unsigned int timer = 0;

	startTimer(&timer);	
	gpu_fx_map_kernel1<<<numBlock, numThread>>>(d_x, d_c, N);
	cutilSafeCall(cudaThreadSynchronize());
	cutilCheckMsg("gpu_fx_map_kernel1");
	endTimer(&timer, "gpu_fx_map_kernel");

	GPUFREE(d_x);
	GPUFREE(d_c);
}


#endif /* __GDD_TEST_CU__ */


