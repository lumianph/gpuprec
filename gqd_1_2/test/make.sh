CUDA_SDK_HOME="/home/mian/NVIDIA_GPU_Computing_SDK"
NVCC_FLAGS="-m64 -gencode arch=compute_35,code=sm_35"
INC="-I/home/xuntao/lumian/local/include"

echo "Compiling GPU kernel ......"
nvcc gqdtest_kernel.cu -c -O3 -I../inc -I ../map -I$CUDA_SDK_HOME/C/common/inc $NVCC_FLAGS $INC

echo "Compiling test cases ......"
g++ test_util.cpp -c -O3 -I ../inc -I /usr/local/cuda/include -fopenmp $INC
g++ benchmark.cpp -c -O3 -I../inc -I/usr/local/cuda/include -fopenmp $INC

echo "Linking ......"
g++ test_util.o gqdtest_kernel.o benchmark.o -o benchmark -O3 -L$CUDA_SDK_HOME/C/lib -L/usr/local/cuda/lib64 -L/home/xuntao/lumian/local/lib  -lqd -lcuda -lcudart -fopenmp
