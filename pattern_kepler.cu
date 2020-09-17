# include <stdio.h>
# include <stdint.h>

# include "cuda_runtime.h"

//compile nvcc *.cu -o test

__global__ void global_latency (unsigned int * my_array, int array_length, int iterations,  unsigned int * duration, unsigned int *index);


void parametric_measure_global(int N, int iterations);

void measure_global();


int main(){

    cudaSetDevice(0);

    measure_global();

    cudaDeviceReset();
    return 0;
}


void measure_global() {

    int N, iterations;
    //stride in element
    iterations = 1;
	
    N = 400*1024*1024;
    printf("\n=====%10.4f MB array, Kepler pattern read, read 160 element====\n", sizeof(unsigned int)*(float)N/1024/1024);
    parametric_measure_global(N, iterations);
    printf("===============================================\n\n");
	
}


void parametric_measure_global(int N, int iterations) {
    cudaDeviceReset();

    cudaError_t error_id;
	
    int i;
    unsigned int * h_a;
    /* allocate arrays on CPU */
    h_a = (unsigned int *)malloc(sizeof(unsigned int) * (N+2));
    unsigned int * d_a;
    /* allocate arrays on GPU */
    error_id = cudaMalloc ((void **) &d_a, sizeof(unsigned int) * (N+2));
    if (error_id != cudaSuccess) {
        printf("Error 1.0 is %s\n", cudaGetErrorString(error_id));
    }

    /* initialize array elements*/
    for (i=0; i<N; i++)
        h_a[i] = 0;
    /*
     * 32 MB (8M) stride access pattern:
     *
     * h[0]=8M,      h[1]=8M+1
     * h[8M]=16M,    h[8M+1]=16M+1
     * ...
     * h[384M]=392M, h[384M+1]=392M+1
     * h[392M]=400M, h[392M+1]=400M+1
     *
     * Stage 1:
     * If we start from j=0 and follow the pointer as
     * j=h[j]
     * then we will visit indice: 0, 8M, 16M,...,392M <--- 49 indices
     *
     * Stage 3:
     * When we get to j=1, we start the 8M stride again
     * 1,8M+1,...,392M+1 <--- 49 indices
     */
    for (i=0; i<50; i++){
        h_a[i * 1024 * 1024 * 8] = (i+1)*1024*1024*8;
        h_a[i * 1024 * 1024 * 8 + 1] = (i+1)*1024*1024*8+1;
    }
    // 1568 MB entry
    /*
     * 4B (1 stride)
     *
     * h[392M+1]=392M+2
     * h[392M+2]=392M+3
     * h[392M+3]=392M+1
     *
     * Stage 4:
     * When we get j=392M+1, we start the 1 stride pattern as
     * 392M+2, 392M+3, 392M+1, 392M+2 ... <--- until the end
     */
    h_a[392*1024*1024+ 1] = 392*1024*1024 + 2;
    h_a[392*1024*1024 + 2] = 392*1024*1024 + 3;
    h_a[392*1024*1024 + 3] = 392*1024*1024 + 1;

    /*
     * 1MB (.25M) stride
     *
     * h[392M]=392M+.25M
     * h[392M+.25N]=392M+.5M
     * ...
     * h[392M+7.5M]=392M+7.75M
     * h[392M+7.75M]=1
     *
     * Stage 2:
     * When we get to j=392M, we keep going as
     * (392+.25)M, (392+.5)M,...,(392+7.75)M. <--- 30 indices
     * Then we have j=h[(392+7.75)M]=1
     */
    for (i=0; i< 31; i++)
        h_a[(i+1568)*1024*256] = (i + 1569)*1024*256;
    h_a[1599*1024*256] = 1;
	

    h_a[N] = 0;
    h_a[N+1] = 0;
    /* copy array elements from CPU to GPU */
    error_id = cudaMemcpy(d_a, h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
    if (error_id != cudaSuccess) {
        printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
    }


    unsigned int *h_index = (unsigned int *)malloc(sizeof(unsigned int)*256);
    unsigned int *h_timeinfo = (unsigned int *)malloc(sizeof(unsigned int)*256);

    unsigned int *duration;
    error_id = cudaMalloc ((void **) &duration, sizeof(unsigned int)*256);
    if (error_id != cudaSuccess) {
        printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
    }


    unsigned int *d_index;
    error_id = cudaMalloc( (void **) &d_index, sizeof(unsigned int)*256 );
    if (error_id != cudaSuccess) {
        printf("Error 1.3 is %s\n", cudaGetErrorString(error_id));
    }

    cudaDeviceSynchronize ();
    /* launch kernel*/
    dim3 Db = dim3(1);
    dim3 Dg = dim3(1,1,1);


    global_latency <<<Dg, Db>>>(d_a, N, iterations,  duration, d_index);

    cudaDeviceSynchronize ();

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        printf("Error kernel is %s\n", cudaGetErrorString(error_id));
    }

    /* copy results from GPU to CPU */
    cudaDeviceSynchronize ();



    error_id = cudaMemcpy((void *)h_timeinfo, (void *)duration, sizeof(unsigned int)*256, cudaMemcpyDeviceToHost);
    if (error_id != cudaSuccess) {
        printf("Error 2.0 is %s\n", cudaGetErrorString(error_id));
    }
    error_id = cudaMemcpy((void *)h_index, (void *)d_index, sizeof(unsigned int)*256, cudaMemcpyDeviceToHost);
    if (error_id != cudaSuccess) {
        printf("Error 2.1 is %s\n", cudaGetErrorString(error_id));
    }

    cudaDeviceSynchronize ();

    for(i=0;i<256;i++)
        printf("%3d: %d\t %d\n", i,h_index[i], h_timeinfo[i]);

    /* free memory on GPU */
    cudaFree(d_a);
    cudaFree(d_index);
    cudaFree(duration);


    /*free memory on CPU */
    free(h_a);
    free(h_index);
    free(h_timeinfo);
	
    cudaDeviceReset();

}



__global__ void global_latency (unsigned int * my_array, int array_length, int iterations, unsigned int * duration, unsigned int *index) {

    unsigned int start_time, end_time;
    unsigned int j = 0;

    __shared__ unsigned int s_tvalue[256];
    __shared__ unsigned int s_index[256];

    int k;

    for(k=0; k<160; k++){
        s_index[k] = 0;
        s_tvalue[k] = 0;
    }

    //first round
//	for (k = 0; k < iterations*256; k++) 
//		j = my_array[j];
	
    //second round
    for (k = 0; k < iterations*256; k++) {
		
        start_time = clock();

        j = my_array[j];
        s_index[k]= j;
        end_time = clock();

        s_tvalue[k] = end_time-start_time;

    }

    my_array[array_length] = j;
    my_array[array_length+1] = my_array[j];

    for(k=0; k<256; k++){
        index[k]= s_index[k];
        duration[k] = s_tvalue[k];
    }
}



