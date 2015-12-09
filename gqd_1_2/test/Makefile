
#########################################################################
# Compiler options

# NVCC compiler
NVCC_HOME       = /usr/local/cuda
NVCC            = $(NVCC_HOME)/bin/nvcc
NVCC_FLAG				= -m64 -gencode arch=compute_35,code=sm_35 --fmad=false

# C++ compiler
GCC							= g++

# Common definition
OPT							= -O3
INC							= -I$(NVCC_HOME)/include -I../inc
DEF							=
LIB							= -fopenmp -L/usr/local/cuda/lib64 -L/home/xuntao/lumian/local/lib -lqd -lcuda -lcudart

# Output objects
OUTPUT					= benchmark
OBJS						

#########################################################################
# Compilation

$(OUTPUT): $(OBJS)
	$(GCC) $(LIB) $(OBJS) -o $(OUTPUT)
