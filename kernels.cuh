#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

__global__ void gpuBigDot(float *aVec, float *bVec, float *dot, size_t vecLen);

float cpuBigDot(float *aVec, float *bVec, size_t vecLen);

#endif







































































// #ifndef __KERNELS_CUH__
// #define __KERNELS_CUH__

// __global__ void dot_product_kernel(float *x, float *y, float *dot, unsigned int n);

// #endif
