#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <dirent.h> 
#include <string.h> 

#define max(a,b) ({ a > b ? a : b; })
#define min(a,b) ({ a < b ? a : b; })

#define MASK (4)
#define THREAD_IN_BLOCK (16)

#define THREAD_IN_BLOCK_GRAY (512)
#define NSTREAMS (4)

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define INPUT ("./img/img.bmp")
#define OUTPUT ("./out/output.bmp")

typedef struct bmp_header{
    unsigned short identifier;      // 0x0000
    unsigned int filesize;          // 0x0002
    unsigned int reserved;          // 0x0006
    unsigned int bitmap_dataoffset; // 0x000A
    unsigned int bitmap_headersize; // 0x000E
    unsigned int width;             // 0x0012
    unsigned int height;            // 0x0016
    unsigned short planes;          // 0x001A
    unsigned short bits_perpixel;   // 0x001C
    unsigned int compression;       // 0x001E
    unsigned int bitmap_datasize;   // 0x0022
    unsigned int hresolution;       // 0x0026
    unsigned int vresolution;       // 0x002A
    unsigned int usedcolors;        // 0x002E
    unsigned int importantcolors;   // 0x0032
    unsigned int palette;           // 0x0036
}__attribute__((packed,aligned(1))) bmp_header; //enforce memory alignment, 1 is for not padding

__constant__ int sobel_x[3][3] =
    { { 1, 0, -1 },
      { 2, 0, -2 },
      { 1, 0, -1 } };

__constant__ int sobel_y[3][3] =
    { { 1, 2, 1 },
      { 0,  0,  0 },
      { -1, -2, -1 } };
	  
__constant__ float gray_value[3] = {0.3, 0.58, 0.11};


__global__ void cuda_gray(unsigned char *input, int offset, int streamSize, unsigned char* gray, int size) {
	
	int gray_idx = (offset/3) + (blockIdx.x * blockDim.x + threadIdx.x);
	int rgb_idx = (offset) + ((blockIdx.x * blockDim.x + threadIdx.x) * 3); 
	
	if (((blockIdx.x * blockDim.x + threadIdx.x)*3)>=streamSize || gray_idx>=size) {
		return;
	}

	gray[gray_idx] = (gray_value[0] * input[rgb_idx]) + (gray_value[1] * input[rgb_idx + 1]) + (gray_value[2] * input[rgb_idx + 2]);
}
	  
__global__ void cuda_sobel(unsigned char* d_gray, unsigned char* result, int height, int width) {
	
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	
	int index;
	int gx, gy;
	int x, y;
	
	for(y=0; y<MASK; ++y) {
		for(x=0; x<MASK; ++x) {
			index = ((row * MASK) + y) * width + ((col*MASK) + x);
			
			if(index>(width*height)) {
				return;
			}
			/*	
			1  2  3
			4  5  6
			7  8  9
			*/
			
			// Border Detection
			// Bottom, top, right, left
			if(index < ((width*height) - width) && index>(width-1) && ((index+2)%width)!=0 && ((index+1)%width)!=0) {
				gx = (d_gray[index - width - 1]) + (sobel_x[1][0] * d_gray[index - 1]) + (d_gray[index + width -1]) + //1 4 7
				 (sobel_x[0][2] * d_gray[index - width + 1]) + (sobel_x[1][2] * d_gray[index + 1]) + (sobel_x[2][2] * d_gray[index + width + 1]); // 3 6 9
				 
				gy = (d_gray[index - width - 1]) + (sobel_y[0][1] * d_gray[index - width]) + (sobel_y[1][0] * d_gray[index - 1]) + (d_gray[index + width +1]) + //1 2 3
				 (sobel_y[2][0] * d_gray[index + width - 1]) + (sobel_y[2][1] * d_gray[index + width]) + (sobel_y[2][2] * d_gray[index + width + 1]); // 7 8 9
			
				result[index] = (unsigned char)min(255.0f, max(0.0f, sqrtf(gx * gx + gy * gy)));
			}
			else {
					result[index] = 0;
			}
		}
	}
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
		exit(code);
	  }
   }
}

int main() {
	
	//Input file
	FILE* img = fopen(INPUT,"r");
	
	//Output file
    FILE* output;
	
	//Pixel to support operations
    unsigned char pixel[3];
	
	//Size of image
    int size;
	
	//To show elapsed time
	clock_t start;
	
	//To inizilize gpu
	cudaFree(0);
	
	start = clock();
	
	//Load header of bitmap
    bmp_header bmp_head;
    fread(&bmp_head, sizeof(bmp_header), 1, img);

	size = bmp_head.width * bmp_head.height;
	//pixel of image
	unsigned char* image_data = (unsigned char*) malloc(size * 3);
	
	// Input for device
	unsigned char* d_image_data;
	
	// Output for device
	unsigned char* d_gray;
	
	//Pinned memory
	//gpuErrchk(cudaMallocHost(&image_data, size * sizeof(unsigned char) * 3));
	
	//Read image
	fread(image_data, sizeof(unsigned char), size * 3, img);
	
	//Memory set for input data
	gpuErrchk(cudaMalloc(&d_image_data, size * sizeof(unsigned char) * 3));
	
	//Memory set for output data (gray image)
	gpuErrchk(cudaMalloc(&d_gray, size * sizeof(unsigned char)));
	
	int streamSize = ((size * 3) / NSTREAMS);
	cudaStream_t streams[NSTREAMS];
	
	for (int i = 0; i < NSTREAMS; ++i) {
	  int offset = i * streamSize;
	  cudaStreamCreate(&streams[i]);
	  cudaMemcpyAsync(&d_image_data[offset], &image_data[offset], streamSize, cudaMemcpyHostToDevice, streams[i]);
	  cuda_gray<<<(streamSize/THREAD_IN_BLOCK_GRAY) + 1, THREAD_IN_BLOCK_GRAY, 0, streams[i]>>>(d_image_data, offset, streamSize, d_gray, size);
	}
	
	//Device memory for sobel result
	unsigned char* d_newColors;
	
	//Set block size
	dim3 block(bmp_head.width/(THREAD_IN_BLOCK*MASK) +1 , bmp_head.height/(THREAD_IN_BLOCK*MASK) + 1);
	
	//Set grid size
	dim3 grid(THREAD_IN_BLOCK, THREAD_IN_BLOCK);
	
	//Allocate device memory for result
	gpuErrchk(cudaMalloc(&d_newColors, size * sizeof(unsigned char)));
	
	for (int c=0; c<NSTREAMS; ++c) {
		cudaStreamSynchronize(streams[c]);
	}
	
	//Launch kernel for sobel filter
    cuda_sobel<<<block, grid>>>(d_gray, d_newColors, bmp_head.height, bmp_head.width);
	
	//Check for occurred error
	gpuErrchk(cudaGetLastError());
	
	//Copy data to host memory
	gpuErrchk( cudaMemcpy(image_data, d_newColors, size * sizeof(unsigned char), cudaMemcpyDeviceToHost) );
	
	printf("Elapsed time: %lf\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);
	
	//Write Output
    output = fopen(OUTPUT, "wb");
    fwrite(&bmp_head, sizeof(bmp_header), 1, output);

	for(int c=0; c<size;++c){
		memset(pixel, image_data[c], sizeof(pixel));
        fwrite(pixel, sizeof(unsigned char) * 3, 1, output);
	}
	   
	//Free memory
	for (int c=0; c<NSTREAMS; ++c) {
		cudaStreamDestroy(streams[c]);
	}
	
	cudaFree(d_image_data);
	cudaFree(d_gray);
	cudaFree(d_newColors);
	free(image_data);
	
    fclose(img);
    fclose(output);
	cudaDeviceReset();
	
	return 0;
}