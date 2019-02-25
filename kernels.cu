#include "kernels.cuh"


/* function to be executed on the GPU and is callable from the host*/
__global__ void gpuBigDot(float *aVec, float *bVec, float *dot, size_t vecLen){
    
    /* Get the global thread ID */
    size_t iVec = blockIdx.x*blockDim.x+threadIdx.x;

    /* size of chunks we take when calculating the sum */
    size_t stride = blockDim.x*gridDim.x;
    __shared__ float cache[1024];

    /* Assures that not more threads assigned than items in array */
    double temp = 0.0;  
    if (iVec < vecLen){
        temp = aVec[iVec] * bVec[iVec];
	iVec += stride;
    }

    cache[threadIdx.x] = temp;

    __syncthreads();

    /* reduces thread count as we sum */
    unsigned int i = blockDim.x/2;
    while(i != 0){
        if(threadIdx.x < i){
            cache[threadIdx.x] += cache[threadIdx.x + i];
	}
	__syncthreads();
	i /= 2;
    }
    if(threadIdx.x == 0){
	atomicAdd(dot, cache[0]);
    }
}

float cpuBigDot(float *aVec, float *bVec, size_t vecLen){
    size_t iVec;
    float result=0.0;
    for(iVec = 0; iVec < vecLen; ++iVec) {
        result += aVec[iVec] * bVec[iVec];
    }
    return result;
}
