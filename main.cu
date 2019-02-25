#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "kernels.cuh"


#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


int main() {

    static size_t vecLength = 1024*1024; /*number of elements in array*/
    size_t iVec;
    
    float *h_aVec, *h_bVec, *h_c;/* Host vectors*/    
    float *d_aVec, *d_bVec, *d_c; /* Device vectors*/

    clock_t startAll, endAll, startCPU, endCPU;
    clock_t startGPUKern, endGPUKern,endCtoG;
    double diffsAll, diffsCPU, diffsKernel;
    double diffsCtoG, diffsGtoC;
    time_t t;
 
    size_t gridSize, blockSize;
    blockSize = 1024;    /* Number of threads in each thread block*/
    gridSize = (int)ceil((float)vecLength/blockSize);  /* Number of thread blocks in grid*/

    
    size_t singleNumber = sizeof(float);   /*percision of number calculated*/    
    size_t bytes = vecLength*singleNumber; /* Size, in bytes, of each vector */
    startAll = clock();
    /* Allocate memory for each vector on host */
    h_aVec = (float *) malloc(bytes);
    h_bVec = (float *) malloc(bytes);
    h_c = (float *) malloc(sizeof(float));

    /* Allocate memory for each vector on device */
    cudaMalloc((void**)&d_aVec, bytes);
    cudaMalloc((void**)&d_bVec, bytes);
    cudaMalloc((void**)&d_c, singleNumber);
    cudaMemset(d_c, 0.0, singleNumber); /* Initialize device value at zero*/
    cudaCheckErrors("cudaMalloc fail");

    srand((unsigned) time(&t)); /* Intializes random number generator */

    /* Intializes host varibles with random variables */
    for(iVec = 0; iVec < vecLength; ++iVec){
        h_aVec[iVec] = rand();
        h_bVec[iVec] = rand();
    }
    endCtoG = clock();
    /* Copy arrays from host to device*/
    cudaMemcpy( d_aVec, h_aVec, bytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_bVec, h_bVec, bytes, cudaMemcpyHostToDevice );
    cudaCheckErrors("cudaMemcpy 1 fail");
    
    startGPUKern = clock();

    /*launch configuration <<<numberOfThreadBlocks,numberOfThreadsWithinEachBlock*/
    gpuBigDot<<<gridSize, blockSize>>>(d_aVec,d_bVec,d_c,vecLength);
    cudaCheckErrors("kernel fail");
    endGPUKern = clock();

    /* Copy value from device back to host*/
    cudaMemcpy(h_c, d_c, singleNumber, cudaMemcpyDeviceToHost );
    cudaCheckErrors("cudaMemcpy 2 fail");
    endAll = clock();



    /*Uncomment To Check the first 10 values*/
    /****************************************************
    size_t numbOfTestVals = 10; 
    size_t i;
    printf("\nindex\t       vector A\t       vector B\t\ta.b\n");
    for(i = 0; i<numbOfTestVals; ++i)
        printf("[%ld]\t%15.2lf\t%15.2lf\t\t%15.2lf\n",i,h_aVec[i],h_bVec[i],h_aVec[i]*h_bVec[i]);    
    ****************************************************/

    printf("a.b GPU value = %lf\n",* h_c);
    startCPU = clock();
    printf("a.b CPU value = %lf\n",cpuBigDot(h_aVec,h_bVec,vecLength));
    endCPU = clock();

    diffsCPU = (endCPU - startCPU)/(double)CLOCKS_PER_SEC;
    diffsKernel = (endGPUKern - startGPUKern)/(double)CLOCKS_PER_SEC;
    diffsAll = (endAll - startAll)/(double)CLOCKS_PER_SEC;
    diffsCtoG = (endCtoG - startAll)/(double)CLOCKS_PER_SEC;
    diffsGtoC = (endAll - endGPUKern)/(double)CLOCKS_PER_SEC;

    printf("error value = %lf%%\n",(* h_c-cpuBigDot(h_aVec,h_bVec,vecLength))/cpuBigDot(h_aVec,h_bVec,vecLength)*100);


    printf("\nCPU: Tcpu = %lf seconds\n",diffsCPU);
    printf("GPU: Tgpu = %lf seconds\n",diffsAll);
    printf("GPU: Memory allocation and data transfer from CPU to GPU = %lf seconds\n",diffsCtoG);
    printf("GPU: Kernel execution = %lf seconds\n",diffsKernel);
    printf("GPU: Data transfer from GPU to CPU time = %lf seconds\n",diffsGtoC);
    printf("Speedup = GPU/CPU = %lf\n",diffsAll/diffsCPU);


    /* Frees the device memory allocated to varibles */
    cudaFree(d_aVec);
    cudaFree(d_bVec);
    cudaFree(d_c);

    /* Frees the host device memory allocated to varibles */
    free(h_aVec);
    free(h_bVec);
    free(h_c);

    return 0;
}
