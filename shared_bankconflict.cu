#include <stdio.h>

#include "repeat.h"
#include "cuda_runtime.h"

__global__ void shared_latency_single_thread (unsigned int * my_array, int array_length, int iterations, unsigned long long * duration);

__global__ void shared_latency_single_block(int * my_array, int array_length, int iterations, unsigned long int * d_latency, int stride);

void shared_banksize_query();

void parametric_measure_shared(int N, int iterations, int stride);

void measure_shared_latency();

void shared_access_latency(int stride);

int main() {

	//shared_banksize_query();

	//single thread to access shared memory latency
	//measure_shared_latency();

	//measure one block shared memory access latency
	printf("======Measure bank conflict latency=====");	
	printf("\nstride\t Fermi\t kepler_4byte\t Kepler_8byte\n");	
	for(int i = 0; i <= 64; i+=2){
		printf("%d\t", i);
		cudaSetDevice(1);
		shared_access_latency(i);
		cudaDeviceReset();
		cudaSetDevice(0);
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
		shared_access_latency(i);
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		shared_access_latency(i);
		cudaDeviceReset();
		printf("\n");
		}
	return 0;

}

__global__ void shared_latency_single_thread (unsigned int * my_array, int array_length, int iterations, unsigned long long * duration) {

   unsigned int start_time, end_time;
   int i, k;
   unsigned int j = 0;
   unsigned long long sum_time;

   my_array[array_length - 1] = 0;
   sum_time = 0;
   duration[0] = 0;


   // sdata[] is used to hold the data in shared memory. Dynamically allocated at launch time.
   extern __shared__ unsigned int sdata[];

   for (i=0; i < array_length; i++) {
      sdata[i] = my_array[i];
   }

   j=0;
   for (k= 0; k<= iterations; k++) {
   	  if (k==1) {
	  	sum_time = 0;
	  }

 	  start_time = clock();
	  repeat256(j=sdata[j];);
	  end_time = clock();

   	  sum_time += (end_time -start_time);
   }

   my_array[array_length - 1] = j;
   duration[0] = sum_time;

}

void shared_banksize_query(){

	/* shared memory banksize configuration*/
	cudaSharedMemConfig pConfig;
	cudaError_t error_id;

	error_id = cudaDeviceGetSharedMemConfig(&pConfig);
	if (error_id != cudaSuccess) {
		printf("Error 2 is %s\n", cudaGetErrorString(error_id));
	}
	
	printf("Shared memory bank size: ");
	switch(pConfig){
			case 0: printf(" default \n");break;
			case 1: printf(" 4 bytes \n");break;
			case 2: printf(" 8 bytes \n");break;
	}
}

// Shared memory array size is N-2. Last two elements are used as dummy variables.
void parametric_measure_shared(int N, int iterations, int stride) {
	
	int i;
	unsigned int * h_a;
	unsigned int * d_a;

	unsigned long long * duration;
	unsigned long long * latency;

	cudaError_t error_id;

	/* allocate array on CPU */
	h_a = (unsigned int *)malloc(sizeof(unsigned int) * N);
	latency = (unsigned long long *)malloc(2*sizeof(unsigned long long));

   	/* initialize array elements on CPU */
	for (i = 0; i < N-2; i++) {
		h_a[i] = (i + stride) % (N-2);	
//		printf("A[%d]= %d\n",i, h_a[i]);
	}
	h_a[N-2] = 0;
	h_a[N-1] = 0;


	/* allocate arrays on GPU */
	cudaMalloc ((void **) &d_a, sizeof(unsigned int) * N);
	cudaMalloc ((void **) &duration, 2*sizeof(unsigned long long));

        cudaThreadSynchronize ();
	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error 1 is %s\n", cudaGetErrorString(error_id));
	}

        /* copy array elements from CPU to GPU */
        cudaMemcpy((void *)d_a, (void *)h_a, sizeof(unsigned int) * N, cudaMemcpyHostToDevice);
        cudaMemcpy((void *)duration, (void *)latency, 2*sizeof(unsigned long long), cudaMemcpyHostToDevice);
        
	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error 2 is %s\n", cudaGetErrorString(error_id));
	}
	
	/* launch kernel*/
	dim3 Db = dim3(1);
	dim3 Dg = dim3(1,1,1);

	//printf("Launch kernel with parameters: %d, N: %d, stride: %d\n", iterations, N, stride); 
	int sharedMemSize =  sizeof(unsigned int) * N ;

	shared_latency_single_thread <<<Dg, Db, sharedMemSize>>>(d_a, N, iterations, duration);

	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error 3 is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();

        cudaMemcpy((void *)h_a, (void *)d_a, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)latency, (void *)duration, 2*sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        cudaThreadSynchronize ();

	/* print results*/


	printf("  %d, %f\n",stride,(double)(latency[0]/(256.0*iterations)));


	/* free memory on GPU */
	cudaFree(d_a);
	cudaFree(duration);
	cudaThreadSynchronize ();

        /*free memory on CPU */
        free(h_a);
        free(latency);


}

void measure_shared_latency(){
	int N, stride; 

	// initialize upper bounds here
	int stride_upper_bound = 1024; 

	printf("Shared memory latency for varying stride.\n");
	printf("stride (bytes), latency (clocks)\n");

	N = 256;
	stride_upper_bound = N;
	for (stride = 1; stride <= stride_upper_bound; stride += 1) {
		parametric_measure_shared(N+2, 10, stride);
	}
}

void shared_access_latency(int stride){

	cudaError_t error_id;

	int N = 2048 ; // array with 1024 elements
	int i;

	//double duration;
	int sharedMemSize =  sizeof(unsigned int) * N;
	int iterations = 10;

	// the grid is constructed with single block of one warp
	dim3 Dg = dim3(1, 1, 1);
	dim3 Db = dim3(32, 1, 1); 

	// host array initialize	
	int * h_a = (int *)malloc(sizeof(int) * N);
	for (i=0; i< N; i++){
		h_a[i] = (stride * Db.x +i)%N;
		}

	// device array allocation
	int * d_a;
	cudaMalloc ((void **) &d_a, sizeof(int) * (N+2));
	// an array to store latency information
	unsigned long int * latency = (unsigned long int *)malloc(sizeof(unsigned long int));
	unsigned long int * d_latency;
	cudaMalloc ((void **) &d_latency, sizeof(unsigned long int) );

	/* copy array elements from CPU to GPU */
        cudaMemcpy(d_a, h_a, sizeof(int) * N, cudaMemcpyHostToDevice);
	
	// launch the kernel
	shared_latency_single_block <<<Dg, Db, sharedMemSize>>>(d_a, N, iterations, d_latency, stride);
	cudaThreadSynchronize ();

	error_id = cudaGetLastError();
        if (error_id != cudaSuccess) {
		printf("Error is %s\n", cudaGetErrorString(error_id));
	}

	/* copy results from GPU to CPU */
	cudaThreadSynchronize ();

        //cudaMemcpy((void *)h_a, (void *)d_a, sizeof(unsigned int) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)latency, (void *)d_latency, sizeof(unsigned long int), cudaMemcpyDeviceToHost);

        cudaThreadSynchronize ();


	printf("   %f\t",(double)(latency[0])/iterations/64.0);
	
	//Free the space
	cudaFree(d_a);
	cudaFree(d_latency);

	free(h_a);
        free(latency);
	
}


__global__ void shared_latency_single_block(int *my_array, int array_length,  int iterations, unsigned long int *d_latency, int stride) {

   unsigned long int start_time, end_time;
   unsigned long int sum = 0;
   int i ;
   int data;
	
	my_array[array_length] = 0;
	my_array[array_length+1] = 0;

   // sdata[] is used to hold the data in shared memory. Dynamically allocated at launch time.
   extern __shared__ unsigned int sdata[];

	// copy data from global to shared
   for (i=0; i < array_length / blockDim.x; i++) {
      sdata[i * blockDim.x + threadIdx.x] = my_array[i * blockDim.x + threadIdx.x];
   }


   for ( i=0;i <= iterations; i++ ) {
	data=threadIdx.x*stride	;
	if(i==1) sum = 0;
	start_time = clock();
	repeat64( data=sdata[data];);
	end_time = clock();
	
	
   	sum += (end_time - start_time);
   }
 
	d_latency[0] = sum;

	for (i=0; i < array_length / blockDim.x; i++) {
      		my_array[i * blockDim.x + threadIdx.x] = sdata[i * blockDim.x + threadIdx.x];
   	}
	
	my_array[array_length] = sum;
	my_array[array_length+1] = data;
   
 }

