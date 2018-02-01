
# CuBin
**Cu**da-accelerated **Bin**ary Matrix Factorization

CuBin is an algorithm composed for my Bachelor thesis. It factorizes a large Boolean matrix of size mxn into two smaller matrices A and B with dimenstions mxk and kxn, where k is a variable between 1 and 32. It provides competitive decomposition errors paired with high speedups in comparison to single core execution.
It is based on a pure random approach: instead of calculating the alternatinv updates done to the factor matrices, random changes are created and tried out. If a better error results, the permutation is accepted, otherwise discarded. 

To compile, enter following line:
>nvvc CuBin_final.cu -std=c++11 -arch=sm_61

Replace the architecture with the one on your available GPU. If you want to also test it against a parallelized version with openMP (very, very slow), also add "-*Xcompiler="-fopenmp*" at the end.

Following *-D* parameters are available:
>DIM_PARAM (default 20) - change factor k

>THREADSPERBLOCK (default 1024) - for smaller matrices below 1024 in width or length to prevent thread idling

>CPU - turn on CPU optimization after GPU

>STARTINITIALIZATION {1,2,3} (default 2) - different densities for the initialization of factor matrices

>PERF - turns on performance measurement mode: to plot error curves, intermediate results are written into csv file

The algorithm runs without any parameters except for the *.data* file. Following parameter changes can be done:
>./a.out  [data file]

>[showing error after that many steps]

>[lines at once changed in kernels]

>[threshold abort condition]

>[maximum gpu iterations]

>[number of changed each line gets without change before aborting]

>[starting temperature for metropolis algorithm]

>[iterations until temperature is reduced]

>[factor the temperature is lowered]

If you are interested in benchmarks or my bachelor thesis, write an e-mail to: adrianlamoth@googlemail.com               
 
