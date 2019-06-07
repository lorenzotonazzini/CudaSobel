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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

#define INPUT ("./img/img1.bmp")
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


__global__ void cuda_gray(unsigned char *input, unsigned char* gray, int size) {
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (idx > size) {
		return;
	}
	
	int rgb_index = idx * 3;
	
	gray[idx] = (gray_value[0] * input[rgb_index]) + (gray_value[1] * input[rgb_index + 1]) + (gray_value[2] * input[rgb_index + 2]);
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
	
    int x, y, img_width, img_height;
	
	//To show elapsed time
	clock_t start;
	
	start = clock();
	
	//Load header of bitmap
    bmp_header bmp_head;
    fread(&bmp_head, sizeof(bmp_header), 1, img);

    img_width = bmp_head.width;
    img_height = bmp_head.height;

	//pixel of image
	unsigned char* image_data;
	
	// Input for device
	unsigned char* d_image_data;
	
	// Output for device
	unsigned char* d_gray;
	
	//Pinned Memory
	cudaMallocHost((void**) &image_data, img_width * img_height * 3);
	
	//Read image
	fread(image_data, sizeof(unsigned char), img_width * img_height * 3, img);
	
	//Memory set for input data
	gpuErrchk(cudaMalloc(&d_image_data, img_width * img_height * sizeof(unsigned char) * 3));
	gpuErrchk(cudaMemcpy(d_image_data, image_data, img_width * img_height * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice));
	
	//Memory set for output data (gray image)
	gpuErrchk(cudaMalloc(&d_gray, img_width * img_height * sizeof(unsigned char)));
	
	//Launch kernel to transform image in gray scale
	cuda_gray<<<(img_width * img_height) / (THREAD_IN_BLOCK * THREAD_IN_BLOCK), THREAD_IN_BLOCK * THREAD_IN_BLOCK>>>(d_image_data, d_gray, img_width * img_height);

	//Device memory for sobel result
	unsigned char* d_newColors;
	
	//Host memory for sobel result
	unsigned char* newColors = (unsigned char*) malloc(img_width * img_height * sizeof(unsigned char));
	
	//Allocate device memory for result
	gpuErrchk(cudaMalloc(&d_newColors, img_width * img_height * sizeof(unsigned char)));
	
	//Set Block size
	dim3 block(THREAD_IN_BLOCK, THREAD_IN_BLOCK);
	
	//Set grid size
	dim3 grid(img_width/(THREAD_IN_BLOCK*MASK) +1 , img_height/(THREAD_IN_BLOCK*MASK) + 1);
	
	//Make sure that other kernel has finished
	cudaDeviceSynchronize();
	
	//Launch kernel for sobel filter
    cuda_sobel<<<grid, block>>>(d_gray, d_newColors, img_height, img_width);
	
	//Check for occurred error
	gpuErrchk(cudaGetLastError());
	
	//Copy data to host memory
	gpuErrchk( cudaMemcpy(newColors, d_newColors, img_width * img_height*sizeof(unsigned char), cudaMemcpyDeviceToHost) );
	
	//Make sure that gpu has finished to work
	cudaDeviceSynchronize();
	
	printf("Elapsed time: %lf\n", ((double) (clock() - start)) / CLOCKS_PER_SEC);
	
	//Write Output
    output = fopen(OUTPUT, "wb");
    fwrite(&bmp_head, sizeof(bmp_header), 1, output);

    for(y=0; y<img_height; ++y) {
        for(x=0; x<img_width; ++x){
            memset(pixel, newColors[(y*img_width) + x], sizeof(pixel));
            fwrite(pixel, sizeof(unsigned char) * 3, 1, output);
        }
    }
	   
	//Free memory
	cudaFreeHost(image_data);
	cudaFree(d_image_data);
	cudaFree(d_newColors);
	cudaFree(d_gray);
	free(newColors);
	   
    fclose(img);
    fclose(output);
	
	return 0;
}


