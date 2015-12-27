
#########################################################################
# Compiler options

# NVCC compiler
NVCC_HOME       = /usr/local/cuda
NVCC            = $(NVCC_HOME)/bin/nvcc
NVCC_FLAGS		= -O3 -m64 -gencode arch=compute_35,code=sm_35 --fmad=false

# C++ compiler
GCC		= g++

# Common definition
GCC_FLAGS		= -O3 -std=c++11
INC				= -I$(NVCC_HOME)/include -I../gqd -I/home/xuntao/lumian/local/include
DEF				=
LIB				= -fopenmp -L/usr/local/cuda/lib64 -L/home/xuntao/lumian/local/lib -lqd -lcuda -lcudart

# Output objects
BENCH			= benchmark
SAMPLE			= sample
BENCH_OBJS		= test_util.o benchmark.o
SAMPLE_OBJS		= sample.o test_util.o

#########################################################################
# Compilation

$(BENCH): $(BENCH_OBJS)
	$(GCC) $(BENCH_OBJS) -o $(BENCH) $(LIB)

$(SAMPLE): $(SAMPLE_OBJS)
	$(GCC) sample.o test_util.o -o $(SAMPLE) $(LIB)

test_util.o: test_util.cpp
	$(GCC) -c $(GCC_FLAGS) $(INC) $(DEF) test_util.cpp

benchmark.o: benchmark.cu
	$(NVCC) -c $(NVCC_FLAGS) $(INC) $(DEF) benchmark.cu

sample.o: sample.cu
	$(NVCC) -c $(NVCC_FLAGS) $(INC) $(DEF) sample.cu

	
#########################################################################
# Make targets
all: $(BENCH) $(SAMPLE)

clean:
	rm -f $(BENCH) $(SAMPLE) *.o
