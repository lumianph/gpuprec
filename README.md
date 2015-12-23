# *gpuprec*: Extended-Precision Libraries on GPUs

## INTRODUCTION

This project develops extended precision libraries for GPGPU (CUDA). It was originally developed during my PhD study  (you can still visit the original page for gpuprec from Google Code: https://code.google.com/p/gpuprec/). Recently I am fixing issues of the libraries to make it run on modern generations of GPUs correctly. I hope you will find it useful to your work.

This project essentially consists of two libraries.

**GQD**: This library has implemented double-double (31 decimal digits) and quad-double (62 decimal digits) precision on the GPU. It basically ports the algorithms from the QD library (http://crd.lbl.gov/~dhbailey/mpdist/qd-2.3.17.tar.gz) to the GPU. Due to built-in vectors available in CUDA, this library is very easy to use. Most operations are implemented using function overriding.

**GARPREC**: This library supports arbitrary precision using a number of native precision numbers. It basically ports the algorithms from the ARPREC library (http://crd.lbl.gov/~dhbailey/mpdist/arprec-2.2.18.tar.gz) to the GPU. Due to the coalesced access adopted on the GPU for the efficiency, it uses a different memory layout from the traditional one. Therefore, necessary modifications are required for your programs employing native precision.

**Note that I have released the GQD only for now. GARPREC is still ongoing to make it workable on modern generations of GPUs.**

## INSTALL

1. The GQD library is tested on a NVIDIA K40 card with CUDA 7.5 installed. It is not guaranteed that it can generate the same results on other GPU architectures.

2. To facilitate the debug and verification, the CPU QD library is highly recommended to be installed (our benchmark and sample cannot run without the QD library). It can be downloaded here: http://crd-legacy.lbl.gov/~dhbailey/mpdist/ , or you will find it in the folder *third* of this repository.

3. There is no installation required. In order to use the library, you only need to put the folder *gqd* under a suitable directory and then include "gqd.cu" in your project. Please look at sample.cu or benchmark.cu in the folder *gqd_test* for more details. Note that it is highly suggested to disable FMAD for the nvcc compiler (--fmad=false) to generate more consistent results as the CPU-based QD library.

4. By default, the compute-intensive functions (atan, acos, asin, sinh, cosh, tanh, asinh, acosh, atanh) for the quad-double precision are disabled. This is because they take very long time to compile the test cases due to the complexity (tens of minutes). If you do need those functions in your work, please enable the definition ALL_MATH. **NOTE that, for the current commitment, you still can call those functions even without ALL_MATH defined, but it will just return 0.0 (I am fixing this issue).**

## SUPPORTED OPERATORS

Basic arithmetic operators (+, -, \*,  /) are all well supported that are able to generate consistent results to that of the CPU QD library. The below list shows other major supported mathematical operators. Please do let me know if your work requires more supporting.

```
comparison (<, >, ==, ...)
sqrt
exp
log
sin
cos
acos
asin
tan
```

## BENCHMARK

*benchmark.cu* is a micro-benchmark for accuracy and performance. Type the below commands for compilation.

```
cd gqd_test
make benchmark
```

## SAMPLE

*sample.cu* illustrates a simple example to use the GQD library. It is self-explanatory. The following commands is used for compilation.

```
cd gqd_test
make sample
```


## ABOUT THE ACCURACY

Most functions are tested extensively and but exhaustively. For basic arithmetic operators, it should be able to generate the same results as that of CPU-based QD. For mathematical operations, such as *exp* and *log*, there may be minor difference between GQD and QD. We consider that this is because the native mathematical functions on the CPU and GPU have different implementation details. You are suggested to investigate the result difference first before using GQD. For your reference, the file *gqd_test\benchmark.log* shows the output of our benchmark.

**NOTE**: The compiler flags essentially affect the accuracy of the library. Please pay special attention to disable FMAD.

```
--fmad=false
```
## KNOWN ISSUES

1. For invalid input numbers, the library currently usually only returns 0.0, instead of NaN or other exceptions. For example, for acos, if the input number abs(a) > 1.0, the library simple returns 0.0, but a NaN by the QD library.
2. It may take very long time (tens of minutes) to compile the code if you enable ALL_MATH

## TODO
1. The previous code of the test cases is pretty messy. I am working on it.
2. Define correct error return code, e.g., NaN.

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
