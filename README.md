# *gpuprec*: Extended-Precision Libraries on GPUs

## INTRODUCTION

This project intends to develop extended precision libraries for GPGPU. It is originally developed during my PhD study  (you can still visit the original page for gpuprec: https://code.google.com/p/gpuprec/). Recently I have fixed some issues of the libraries running on modern generations of GPUs. I hope you will find it useful for your research or development.

This project consists of two libraries.

**GQD**: This library has implemented double-double (31 decimal digits) and quad-double (62 decimal digits) precision on the GPU. Due to built-in vectors available in CUDA, this library has a very good portability and is also easy to use. Most operations are implemented using function overriding. The algorithms mainly refer to the QD library on the CPU.

**GARPREC**: This library supports arbitrary precision using varied number of native precision numbers. Due to the coalesced access adopted on the GPU for the efficiency, it uses a different memory layout from the traditional one. Therefore, corresponding modifications are necessary compared with the native precision. The algorithms are from the ARPREC library on the CPU.

I have released GQD for now. GARPREC is still ongoing to make it workable on modern generations of GPUs.

## INSTALL

1. The GQD library is tested on a NVIDIA K40 GPU with CUDA 7.5. It is not guaranteed that it can generate the same result on other GPU archtectures, such as Fermi or earlier generations.

2. To faciliate the debug and verification, the CPU QD library is highly recommended to be installed. It can be downloaded here: http://crd-legacy.lbl.gov/~dhbailey/mpdist/ , or you can find it in the folder *third* of this repository.

3. There is no any complication or installation for this library. In order to use the library, you only need to include the file "gqd.cu" and put the folder "gqd" at a specific path. Please look at sample.cu and Makefile under the folder gqd_test for more details. Note that it is suggested to disable FMAD for the nvcc compiler (--fmad=false) to generate more consistent results as the CPU-based QD library.

## SUPPORTED OPERATORS

Basic arithmetic operators (+, -, \*,  /) are all well supported. The below list shows the major supported mathematical operators. Please let me know if your work requires more supporting.

```
comparison (<, >, ==, ...)
sqrt
exp
log
sin
cos
tan
```

## BENCHMARK

*benchmark.cu* is an accuracy and performance micro-benchmark to test most supported operators. Type the below commands to compile the executable files.

```
cd gqd_test
make benchmark
```

## SAMPLE

*sample.cu* shows a simple example of how to utilize the GQD library. It is quite self-explanatory. You can read it for more details. The following commands will compile the executable file.

```
cd gqd_test
make sample
```


## ABOUT THE ACCURACY

Most functions are tested extensively and but exhaustively. For basic arithmetic operators, it should be able to generate the same results as that of CPU-based QD. For mathematical operations, such as *exp* and *log*. There may be minor difference between the results of GQD and QD. We consider that this is because the native math functions on the CPU and GPU have different implementation details. You may investigate the result difference first before using GQD. For your reference, the file *gqd_test\benchmark.log* shows the output of our benchmark with accuracy test.

**NOTE**: The compiler flags essentially affect the accuracy of the library. Please pay special attention to the nvcc flag 

```
--fmad=false
```

## CITATION
You can cite this library as:

**Mian Lu, Bingsheng He, and Qiong Luo. Supporting extended precision on graphics processors. In Proceedings of the Sixth International Workshop on Data Management on New Hardware (DaMoN), 2010**

```
BibTex:
@inproceedings{Lu:2010:SEP:1869389.1869392,
 author = {Lu, Mian and He, Bingsheng and Luo, Qiong},
 title = {Supporting Extended Precision on Graphics Processors},
 booktitle = {Proceedings of the Sixth International Workshop on Data Management on New Hardware},
 series = {DaMoN '10},
 year = {2010},
 isbn = {978-1-4503-0189-3},
 location = {Indianapolis, Indiana},
 pages = {19--26},
 numpages = {8},
 url = {http://doi.acm.org/10.1145/1869389.1869392},
 doi = {10.1145/1869389.1869392},
 acmid = {1869392},
 publisher = {ACM},
 address = {New York, NY, USA},
} 
```

## CONTACT
```
Dr. Mian Lu:
lumianph@gmail.com
```