#include <stdio.h>
#include <stdlib.h>

//declare the texture
texture<int, 1, cudaReadModeElementType> tex_ref;

/*
 * We are going to traverse the array with a stride of _stride_
 * So the number of accesses in the array is N/_stride_,
 * which is the iterations of the P-chase loop.
 *
 * N is defined according to the cache size of the GPU.
 * stride is defined according to the stage:
 *   1 if determining cache size C and cache line size b
 *   b if determining cache set
 * ITER is defined as N/stride. Note that ITER will be used to
 * allocate shared memory. There are two methods:
 *   1. static shared memory
 *      In this method, ITER must be known at compile time.
 *      Therefore, __CUDA_ARCH__ is used to determine N and ITER
 *      according to the hardware CC.
 *      The author of the paper used 4096 as the ITER, i.e. they used
 *      a fixed number of iterations regardless of array size.
 *      This works when 4096 is larger than N/stride.
 *   2. Dymanic shared memory
 *      In this method, ITER can be computed by N, which can be
 *      defined either by __CUDA_ARCH__ or cudaDeviceProperty::major.
 *      But then we need to specify the shared memory size when
 *      launching the kernel and divided the shared memory into
 *      s_tvalue and s_value inside the kernel manually.
 *      E.g.
 *      texture_latency<<<1,1, ITER*2*sizeof(int)>>>( ... )
 *      Inside the kernel:
 *      extern __shared__ int s[];
 *      int *s_tvalue = s;
 *      int *s_value=&s_tvalue[ITER];
 *
 * We also need to pay attention to the size of shared memory.
 * The default shared memory size on Kepler (cc=3.5) and Pascal (cc=6.1)
 * is 48 KB, i.e. 12288 integers. When it is divided into 2 integer arrays,
 * each array can have maximum of 6144 integers, i.e. the maximum supported
 * ITER is 6144.
 * On Kepler, which has a texture cache of size 12 KB, shared memory is
 * large enough to hold tvalues and values for each iteration.
 * However, on Maxwell and Pascal, the shared memory is not able to hold
 * all values for each iteration.
 *
 * The above problem can be solved by warming up the cache first before
 * recording s_tvalue and s_value. Since the cache is warmed up, we only
 * need to go a few iterations to see the pattern, i.e. ITER = 128.
 *
 * Shared Memory (KB/SM)
 * ===================================================================================
 * Fermi           48/16 or 16/48 configurable with L1
 * Kepler(cc3.5)   48/16 or 16/48 or 32/32 configurable with L1
 * Kepler(cc3.7)   112/16 or 96/32 or 80/48 configurable with L1
 * Maxwell         96 dedicated shared memory
 * Pascall         64 dedicated shared memory (But one block can use up to 32 KB only)
 * Volta           96/32, 80/48, 64/64 configurable with L1
 * Turing
 * Ampere
 * ===================================================================================
 *
 */
// #if __CUDA_ARCH__ >= 500
// #define  N 6144
// #define  ITER 6144
// #else
// #define  N 3072
// #define  ITER 3072
// #endif

#define ITER 128

__global__ void texture_latency (int * my_array,
                                 int size,
                                 unsigned int *duration,
                                 int *index,
                                 int iter /* used to warm up the cache*/
    ) {

    // extern __shared__ int s[];
    // int *s_tvalue = s;
    // int *s_value=&s_tvalue[iter];

    const int it =  ITER;


   __shared__ unsigned int s_tvalue[it];
   __shared__ int s_value[it];

    unsigned int start, end;
    int i,j;

    //initilize j
    j=0;

    // for (i=0; i< iter; i++) {
    //     s_value[i] = -1;
    //     s_tvalue[i]=0;
    // }

    /*
     * Try to load the data first to avoid cold cache miss
     * Note that to warm up the cache, we need to traverse the whole array.
     */
    for (int cnt=0; cnt < iter; cnt++){
        j=tex1Dfetch(tex_ref, j);
    }
    /*
     * Since cold cache miss is avoided, the cache structure can
     * be explored now.
     */
    for (int cnt=0; cnt < it; cnt++) {
			
        start=clock();
        j=tex1Dfetch(tex_ref, j);
        s_value[cnt] = j;
			
        end=clock();
        s_tvalue[cnt] = (end -start);
    }

    for (i=0; i< it; i++){
	duration[i] = s_tvalue[i];
	index[i] = s_value[i];
    }

    my_array[size] = i;
    my_array[size+1] = s_tvalue[i-1];
}



void parametric_measure_texture(int N, int stride) {
    // iterations=stride=1
    // N is the array size

    cudaError_t error_id;

    int * h_a, * d_a;
    int size =  (N+2) * sizeof(int);
    h_a = (int *)malloc(size);
    //initialize array
    for (int i = 0; i < N; i++) {
        h_a[i] = (i + stride) % N;
    }
    h_a[N] = 0;
    h_a[N+1] = 0;
    cudaMalloc ((void **) &d_a, size);
    //copy it to device array
    cudaMemcpy((void *)d_a, (void *)h_a, size, cudaMemcpyHostToDevice);

    // here to change the iteration numbers
    /*
     * We are going to traverse the array with a stride of _stride_
     * So the number of accesses in the array is N/_stride_,
     * which is the iterations of the P-chase loop.
     */
    /*
     * iterations is only used to traverse the array for warm up purpose
     * After warming up, we only keep ITER values.
     */
    int iterations = N/stride;
    int iter = ITER;

    // the time ivformation array and index array
    unsigned int *h_duration = (unsigned int *)malloc(iter*sizeof(unsigned int));
    int *h_index = (int *)malloc(iter*sizeof(int));
	
    int *d_index;
    error_id = cudaMalloc(&d_index,  iter*sizeof(int));
    if (error_id != cudaSuccess) {
        printf("Error 1.1 is %s\n", cudaGetErrorString(error_id));
    }

    unsigned int *d_duration;
    error_id = cudaMalloc(&d_duration,  iter*sizeof(unsigned int));
    if (error_id != cudaSuccess) {
        printf("Error 1.2 is %s\n", cudaGetErrorString(error_id));
    }


    //bind texture
    cudaBindTexture(0, tex_ref, d_a, size );

    cudaDeviceSynchronize ();

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        printf("Error 2 is %s\n", cudaGetErrorString(error_id));
    }

    //for (int l=0; l < 20; l++) {

    // launch kernel
    dim3 Db = dim3(1);
    dim3 Dg = dim3(1,1,1);
    // texture_latency <<<Dg, Db, iter*2*sizeof(int)>>>(d_a, size, d_duration, d_index, iterations);
    texture_latency <<<Dg, Db>>>(d_a, size, d_duration, d_index, iterations);

    cudaDeviceSynchronize ();

    error_id = cudaGetLastError();
    if (error_id != cudaSuccess) {
        printf("Error 3 is %s\n", cudaGetErrorString(error_id));
    }

    cudaDeviceSynchronize ();

    /* copy results from GPU to CPU */
    cudaMemcpy((void *)h_index, (void *)d_index, iter*sizeof(int) , cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)h_duration, (void *)d_duration, iter*sizeof(unsigned int) , cudaMemcpyDeviceToHost);

    //}

    //print the result
    //printf("\n=====Visting the %f KB array, loop %d*%d times======\n", (float)(N)*sizeof(int)/1024.0f, iter, 1);
    for (int i=0;i<iter;i++){
        printf("%4d %10d\t %10f\n", i, h_index[i], (float)h_duration[i]);
    }


    //unbind texture
    cudaUnbindTexture(tex_ref);

    //free memory on GPU
    cudaFree(d_a);
    cudaFree(d_duration);
    cudaFree(d_index);
    cudaDeviceSynchronize ();
	
    // free memory on CPU
    free(h_a);
    free(h_duration);
    free(h_index);
	
}

int main(int argc, char *argv[]) {

    int device = 0;
    cudaSetDevice(device); // 0 for Kepler, 1 for Fermi
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);


    int stride, N;

    /*
     * Texture cache size (#integers) in each cc version
     */
    //int textureSize[7]={0,0,3072,3072,0,6144,6144};
    //txSize = textureSize[deviceProp.major];
    N = atoi(argv[1]);
    stride = atoi(argv[2]);

    //printf("\"%s\": cc=%d.%d, texture cache size=%d KB, shared memory=%ld KB\n",
    //       deviceProp.name, deviceProp.major, deviceProp.minor, N*4/1024, deviceProp.sharedMemPerBlock/1024);

    /*
     * The texture L1 data cache is
     * Fermi:   12 KB ==> 3072 integers
     * kepler:  12 KB ==> 3072 integers
     * Maxwell: 24 KB ==> 6144 integers
     * Pascal:  24 KB? ==> 6144 integers
     *
     * 1. To determine the cache size, N should start with a small value
     *    and increase to 3072. The first cache miss should appear when
     *    N=3073 ===> the cache size is 3072 integers = 12 KB
     *
     * 2. To determine the cache line size, N starts with 3073.
     *    The cache miss rate should stay close when 3073 <= N <= 3072+b
     *    So if we increase N gradually from 3073 and the first time we
     *    see a sudden increase on the cache miss, say N=3081, we can infer
     *    the cache line size b = 3081-3073 = 8 integers = 32 B
     */

    // stage1: overflow with 1 element
    //stride = 1; // in element
    //for (N = 6145; N <=6145; N += stride) {
    parametric_measure_texture(N, stride);
    //}
/*
// stage2: overflow with cache lines
stride = 8; // in element
for (N = 3072; N <=3200; N += stride) {
parametric_measure_texture(N+2, iterations, stride);
}
*/

    cudaDeviceReset();
    //printf("\"%s\": cc=%d.%d, texture cache size=%d KB, shared memory=%ld KB\n",
    //       deviceProp.name, deviceProp.major, deviceProp.minor, N*4/1024, deviceProp.sharedMemPerBlock/1024);
    return 0;

}
