
#########################################################################
# Compiler options

# NVCC compiler
NVCC_HOME       = /usr/local/cuda
NVCC            = $(NVCC_HOME)/bin/nvcc
NVCC_FLAGS	= -m64 -gencode arch=compute_35,code=sm_35 --fmad=false

# C++ compiler
GCC		= g++

# Common definition
OPT		= -O3
INC		= -I$(NVCC_HOME)/include -I../inc
DEF		=
LIB		= -fopenmp -L/usr/local/cuda/lib64 -L/home/xuntao/lumian/local/lib -lqd -lcuda -lcudart

# Output objects
OUTPUT		= benchmark
OBJS		= gqdtest_kernel.o test_util.o benchmark.o

#########################################################################
# Compilation

$(OUTPUT): $(OBJS)
	$(GCC) $(LIB) $(OBJS) -o $(OUTPUT)

gqdtest_kernel.o: gqdtest_kernel.cu
	$(NVCC) -c $(NVCC_FLAGS) $(OPT) $(INC) $(DEF) gqdtest_kernel.cu 

test_util.o: test_util.cpp
	$(GCC)  -c $(OPT) $(INC) $(DEF) test_util.cpp

benchmark.o: benchmark.cpp
	$(GCC)  -c $(OPT) $(INC) $(DEF) benchmark.cpp
	
#########################################################################
# Cleanup

clean:
	rm -f $(OUTPUT) $(OBJS)
