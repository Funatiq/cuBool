﻿# cuBool
**CU**DA-accelerated **Bool**ean Matrix Factorization

enhancements of https://github.com/alamoth/CuBin by Adrian Lamoth

Boolean Matrix Factorization (BMF) is a commonly
used technique in the field of unsupervised data analytics. The
goal is to decompose a ground truth matrix C of shape m × n
into a product of two matrices A and B being either an exact
or approximate rank k factorization of C.

cuBool is based on alternately adjusting rows and columns
of A and B using thousands of lightweight CUDA threads. The
massively parallel manipulation of entries enables full usage of
all available cores on modern CUDA-enabled GPUs. Additionally,
modelling up to 32 consecutive entries of the Boolean matrices A,
B and C as 32-bit integer results in fewer data accesses and faster
computation of inner products. This bit-parallel approach allows
for a significant decrease of memory requirements in contrast
to gradient-based continuous updates of entries on dense repre-
sentations.


Install
-------
cuBool was tested with CUDA 9.2 and g++-7. Before using `make`, be sure to adjust the NVCC `-arch` flag in the Makefile to your GPU achitecture.
To compile GPU version:
```
make cuBool
```
To compile CPU version (can be compiled without CUDA):
```
make cuBool_cpu
```

How to use
----------
Call `./cuBool` or `./cuBool_cpu` without parameters to see all parameter options.

Required structure of dataset file:
- First line defines matrix: `<rows> <column> <number of nonzeros>`
- Every other line defines coordinates of a single entry: `<row> <column>`