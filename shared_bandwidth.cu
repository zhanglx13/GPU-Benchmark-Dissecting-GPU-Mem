#include <stdio.h>

#include "repeat.h"
#include "cuda_runtime.h"


#define SM_BASE	16  //the SM number of the device

#define ILP 8

#define OVERHEAD 36  //the overhead (latency cycles) of a clock() and a suncthreads()

#define SMEM_SIZE 4096 //SMEM_SIZE=threadNum*ILP


__noinline__ __device__ unsigned int get_smid(void)
{
	unsigned int ret;
	asm("mov.u32 %0, %smid;":"=r"(ret) );
	return ret;
}

__global__ void 
	KernelReadSharedMem(unsigned int *duration, 
	int *data,
	unsigned long int *data2){
		__shared__ int sData_1[SMEM_SIZE];
		__shared__ int sData_2[SMEM_SIZE];
		int i;
		for (i=threadIdx.x; i<SMEM_SIZE; i+=blockDim.x){
			sData_1[i] = get_smid();
			sData_2[i] = -1;
		}

		int tid2 = threadIdx.x + blockDim.x;
		int tid3 = tid2 + blockDim.x;
		int tid4 = tid3 + blockDim.x;
		int tid5 = tid4 + blockDim.x;
		int tid6 = tid5 + blockDim.x;
		int tid7 = tid6 + blockDim.x;
		int tid8 = tid7 + blockDim.x;
	
		int reg_1, reg_2, reg_3, reg_4, reg_5, reg_6, reg_7, reg_8;
		int index = threadIdx.x + blockDim.x*blockIdx.x;

		clock_t startTime, endTime;

		//timing process
		startTime = clock();

		reg_1 = sData_1[threadIdx.x] ;
		reg_2 = sData_1[tid2] ;
		reg_3 = sData_1[tid3] ;
		reg_4 = sData_1[tid4] ;
		reg_5 = sData_1[tid5] ;
		reg_6 = sData_1[tid6] ;
		reg_7 = sData_1[tid7] ;
		reg_8 = sData_1[tid8] ;

		sData_2[threadIdx.x] = reg_1 ;
		sData_2[tid2] = reg_2 ;
		sData_2[tid3] = reg_3 ;
		sData_2[tid4] = reg_4 ;
		sData_2[tid5] = reg_5 ;
		sData_2[tid6] = reg_6 ;
		sData_2[tid7] = reg_7 ;
		sData_2[tid8] = reg_8 ;
	
		__syncthreads();
		endTime = clock();

		data2[index] = startTime;
		duration[index] = endTime - startTime;
		data[index] = sData_2[threadIdx.x] + sData_2[tid2]
					+ sData_2[tid3] + sData_2[tid4]
					+ sData_2[tid5] + sData_2[tid6]
					+ sData_2[tid7] + sData_2[tid8]
					;
		
}

void ReadSharedMem(int threadNum, int blockNum){

	cudaSetDevice(0);
	cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );

	unsigned int * h_duration, *d_duration;
	unsigned long int *h_timeStamp, *d_timeStamp;
	int * h_data, *d_data;
	
	
	//grid dimension
	dim3 Db = dim3(threadNum,1,1);
	dim3 Dg = dim3(blockNum,1,1);
	int i;
	int N = Dg.x * Db.x;
	
	//allocate memory space
	h_duration = (unsigned int *) malloc(sizeof(unsigned int) * N);
	h_timeStamp = (unsigned long int *) malloc(sizeof(unsigned long int) * N);
	h_data = (int *) malloc(sizeof(int) * N);

	for (i=0; i<N; i++){
		h_duration[i] = 0;
		h_timeStamp[i] = 0;
	}
	cudaMalloc( (void **) &d_duration, sizeof( unsigned int)*N );
	cudaMalloc( (void **) &d_timeStamp, sizeof(unsigned long int)*N );
	cudaMalloc( (void **) &d_data, sizeof(int)* N );


	//copy the data to device
	cudaMemcpy(
		(void*) d_duration, 
		(void*) h_duration, 
		sizeof(unsigned int) * Db.x * Dg.x, 
		cudaMemcpyHostToDevice);
	cudaMemcpy(
		(void*) d_timeStamp, 
		(void*) h_timeStamp, 
		sizeof(unsigned long int) * Db.x * Dg.x, 
		cudaMemcpyHostToDevice);

	//calling kernel function	
	KernelReadSharedMem<<<Dg, Db>>>(d_duration, d_data, d_timeStamp);
	cudaThreadSynchronize ();

	//copy back result
	cudaMemcpy(
		(void*) h_duration, 
		(void*) d_duration, 
		sizeof(unsigned int) * Db.x * Dg.x, 
		cudaMemcpyDeviceToHost);
	cudaThreadSynchronize ();
	cudaMemcpy(
		(void*) h_timeStamp, 
		(void*) d_timeStamp, 
		sizeof(unsigned long int) * Db.x * Dg.x, 
		cudaMemcpyDeviceToHost);
	cudaThreadSynchronize ();
	cudaMemcpy(
		(void*) h_data, 
		(void*) d_data, 
		sizeof(int) * Db.x * Dg.x, 
		cudaMemcpyDeviceToHost);
	cudaThreadSynchronize ();

	//print result, SM id, latency
	///int m = Dg.x/SM_BASE;
	unsigned long maxStamp = 0, minStamp = ULONG_MAX;
	int j;
	for(j = 0; j < 16; j++) {
		maxStamp = 0;
		minStamp = ULONG_MAX;
		int warpNum = 0;
		for(i=0; i<N; i+=32){
			if (h_data[i]/ILP == j){
				printf("%d\t%d\t%d\t%d\t %u\t%u \n",
						Dg.x, Db.x, h_data[i]/ILP, 
						h_duration[i],
						//1.28*8.0*32/(double)h_duration[i],
						h_timeStamp[i],
						h_duration[i]+h_timeStamp[i]);
				if(h_timeStamp[i] < minStamp)
					minStamp = h_timeStamp[i];
				if(h_timeStamp[i] + h_duration[i] > maxStamp)
					maxStamp = h_timeStamp[i] + h_duration[i];
				warpNum++;
			}
		}

			
		printf("%d %d %d %f\n", j, threadNum, blockNum, 8*1.279*ILP*warpNum*32/(maxStamp-minStamp-OVERHEAD));	
	}

	//free the memory space
	cudaFree(d_duration);cudaFree(d_data);cudaFree(d_timeStamp);
	free(h_duration); free(h_data); free(h_timeStamp);
	cudaDeviceReset();
}

int main(){

	//printf("Shared Memory Throughput:\n");
	//printf("BlockNum\t ThreadNum\t SM_id \tThroughputPerWarp(GB/s)\t TimeStamp\n");
	printf("BlockNum\tThreadNum\tSM_id\tDuration\tstart\tend\n");
	int blockNum, i, currentThreadNum;
	int threadNum[6] ={32, 64, 128, 256 ,512};
	for(i=0; i<5; i++){
		currentThreadNum = threadNum[i];
	for (blockNum=SM_BASE; blockNum <= 6*SM_BASE ; blockNum+=SM_BASE){
		
			ReadSharedMem(currentThreadNum, blockNum);
		//printf("\n\n");
	}
	}
	return 0;
}
