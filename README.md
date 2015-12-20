# *gpuprec*: Extended-Precision Libraries on GPUs

## INTRODUCTION

This project intends to develop extended precision libraries for GPGPU. It is originally developed during my PhD work (you still can visit the old page for gpuprec: https://code.google.com/p/gpuprec/). Recently I have fixed some issues of the libraries running on modern generations of GPUs. I hope you will find it useful for your research or development.

This project consists of two projects.

**GQD**: This library has implemented double-double (31 decimal digits) and quad-double (62 decimal digits) precision on the GPU. Due to built-in vectors available in CUDA, this library has a very good portability and is also easy to use. Most operations are implemented using function overriding. The algorithms mainly refer to the QD library on the CPU.

**GARPREC**: This library supports arbitrary precision using varied number of native precision numbers. Due to the coalesced access adopted on the GPU for the efficiency, it uses a different memory layout from the traditional one. Therefore, corresponding modifications are necessary compared with the native precision. The algorithms are from the ARPREC library on the CPU.

I have released GQD for now. GARPREC is still in progres to make it workable on modern generations of GPUs.

## INSTALL

## HOW-TO


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