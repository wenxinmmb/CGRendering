#include <stdio.h>
#include <math.h>
// A macro for checking the error codes of cuda runtime calls
#define CUDA_ERROR_CHECK(expr) \
  {                            \
    cudaError_t err = expr;    \
    if (err != cudaSuccess)    \
    {                          \
      printf("CUDA call failed!\n%s\n", cudaGetErrorString(err)); \
      exit(1);                 \
    }                          \
  }


__global__
void swapChannel_kernel(uchar3 * device_inputImage, uchar3 * device_outputImage, int rows, int cols)
{
  int idx = blockIdx.x * 512+ threadIdx.x; 
  device_outputImage[idx].y = device_inputImage[idx].x;
  device_outputImage[idx].x = device_inputImage[idx].y;
  device_outputImage[idx].z = device_inputImage[idx].z;

}

__global__
void blurImage_kernel(uchar3 * device_inputImage, uchar3 * device_outputImage, int rows, int cols)
{
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  float colorx = 0;
  float colory = 0;
  float colorz = 0;  

  int margin_up = blockIdx.x; 
  int margin_down = 511 - blockIdx.x;
  int margin_right = 511 - threadIdx.x;
  int margin_left = threadIdx.x;
  int left=-4 ,right=4 ,up=-4 ,down=4;

  if(margin_left >= 4 && margin_right >= 4 && margin_up >= 4 && margin_down >= 4 ){

          for(int i = -4; i < 5 ; i++){
            for(int j = -4; j < 5 ; j++){
            colorx += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].x;
            colory += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].y;
            colorz += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].z;
            }
          }

        colorx = colorx/81.0;
        colory = colory/81.0;
        colorz = colorz/81.0;
  }

  else{
  if(margin_down < 4)
    down = margin_down;

  if(margin_right < 4)
    right = margin_right;

  if(margin_left < 4)
    left = 0-margin_left;

  if(margin_up< 4)
    up = 0-margin_up;

          for(int i = up; i <= down ; i++){
            for(int j = left; j <= right ; j++){
            colorx += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].x;
            colory += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].y;
            colorz += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].z;
            }
          }
        colorx /= (down-up+1)*(right-left+1);
        colory /= (down-up+1)*(right-left+1);
        colorz /= (down-up+1)*(right-left+1);

  }

  device_outputImage[idx].x = colorx;
  device_outputImage[idx].y = colory;
  device_outputImage[idx].z = colorz;
}

__global__
void inplaceFlip_kernel(uchar3 * device_inputImage, int rows, int cols)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ uchar3 sharedmem[512];
    sharedmem[threadIdx.x] = device_inputImage[idx];


  __syncthreads();
    device_inputImage[blockIdx.x * blockDim.x + 511 - threadIdx.x] = sharedmem[threadIdx.x] ;


}

__global__
void creative_kernel(uchar3 * device_inputImage, uchar3 * device_outputImage, int rows, int cols)
{

//using algorithms founded online

//44655 , 16900

  int idx = blockIdx.x * 512 + threadIdx.x; 
  int radian = 140;
  int blkcen = 295;
  int threadcen = 335;
  float distance = (blkcen - blockIdx.x) * (blkcen - blockIdx.x) + (threadcen - threadIdx.x) * (threadcen - threadIdx.x);
  distance = sqrt(distance);

  if(distance > radian ){
  float colorx = 0;
  float colory = 0;
  float colorz = 0; 

  float level = distance/10; 
  int intensity = 5;

  if(level <18)
   intensity = 0;
  else if(level < 25)
   intensity = 1;
  else if(level < 30)
   intensity = 2;

  else if(level < 33)
   intensity = 3;
  else if(level < 35)
   intensity = 4;



  int margin_up = blockIdx.x; 
  int margin_down = 511 - blockIdx.x;
  int margin_right = 511 - threadIdx.x;
  int margin_left = threadIdx.x;
  int left= 0-intensity ,right=intensity ,up=0 - intensity ,down=intensity;

  if(margin_left >= intensity && margin_right >= intensity && margin_up >= intensity && margin_down >= intensity){

          for(int i = 0-intensity; i < intensity+1 ; i++){
            for(int j = 0-intensity; j < intensity+1 ; j++){
            colorx += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].x;
            colory += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].y;
            colorz += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].z;
            }
          }

        colorx = colorx/((2*intensity+1)*(2*intensity+1));
        colory = colory/((2*intensity+1)*(2*intensity+1));
        colorz = colorz/((2*intensity+1)*(2*intensity+1));
  }

  else{
  if(margin_down < intensity)
    down = margin_down;

  if(margin_right < intensity)
    right = margin_right;

  if(margin_left < intensity)
    left = 0-margin_left;

  if(margin_up< intensity)
    up = 0-margin_up;

          for(int i = up; i <= down ; i++){
            for(int j = left; j <= right ; j++){
            colorx += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].x;
            colory += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].y;
            colorz += device_inputImage[(blockIdx.x+i)*512+(threadIdx.x+j)].z;
            }
          }
        colorx /= (down-up+1)*(right-left+1);
        colory /= (down-up+1)*(right-left+1);
        colorz /= (down-up+1)*(right-left+1);

  }

  device_outputImage[idx].x = colorx;
  device_outputImage[idx].y = colory;
  device_outputImage[idx].z = colorz;
  }

else{
float Gray = (device_inputImage[idx].z * 0.299 + device_inputImage[idx].y * 0.587 + device_inputImage[idx].x * 0.114);
device_outputImage[idx].x = device_outputImage[idx].y = device_outputImage[idx].z = Gray; }

}


__host__
float filterImage(uchar3 *host_inputImage, uchar3 *host_outputImage, int rows, int cols, int filterNumber){
  
   cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));

  int numPixels = rows * cols;

  //allocate memory on device (GPU)
  uchar3 *device_inputImage;
  uchar3 *device_outputImage;

  CUDA_ERROR_CHECK(cudaMalloc(&device_inputImage, sizeof(uchar3) * numPixels));
  CUDA_ERROR_CHECK(cudaMalloc(&device_outputImage, sizeof(uchar3) * numPixels));
  CUDA_ERROR_CHECK(cudaMemset(device_outputImage, 0,  sizeof(uchar3) * numPixels)); //make sure no memory is left laying around

  //copy input image to the device (GPU)
  CUDA_ERROR_CHECK(cudaMemcpy(device_inputImage, host_inputImage, sizeof(uchar3) * numPixels, cudaMemcpyHostToDevice));

  //start timing to measure length of kernel call
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  //----------------------------------------------------------------
  // TODO: Fill in the parameters for the kernel calls
  //----------------------------------------------------------------
  // Each of the parameters are as follows:
  //    1. Number of thread blocks, can be either int or dim3 (see CUDA manual)
  //    2. Number of threads per thread block, can be either int or dim3 (see CUDA manual)

  // Also note that you pass the pointers to the device memory to the kernel call

  dim3 gridSize = 512;
  dim3 blockSize = 512;

  
  switch(filterNumber){
    case 1:
      swapChannel_kernel<<<gridSize,blockSize>>>(device_inputImage, device_outputImage, rows, cols);
      break;
    case 2:
      blurImage_kernel<<<gridSize,blockSize>>>(device_inputImage, device_outputImage, rows, cols);
      break;
    case 3:
      inplaceFlip_kernel<<<gridSize,blockSize>>>(device_inputImage, rows, cols);
      break;
    case 4:
      creative_kernel<<<gridSize,blockSize>>>(device_inputImage, device_outputImage, rows, cols);
      break;
    default:
      break;
  }

  //----------------------------------------------------------------
  // END KERNEL CALLS - Do not modify code beyond this point!
  //----------------------------------------------------------------

  //stop timing
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);

  float timeElapsedInMs = 0;
  cudaEventElapsedTime(&timeElapsedInMs, start, stop);

  //synchronize
  cudaDeviceSynchronize(); CUDA_ERROR_CHECK(cudaGetLastError());

  //copy device output image back to host output image
  //special case for filter swap - since it is in place, we actually copy the input image back to the host output
  if (filterNumber==3){
    CUDA_ERROR_CHECK(cudaMemcpy(host_outputImage, device_inputImage, sizeof(uchar3) * numPixels, cudaMemcpyDeviceToHost));
  }else{
    CUDA_ERROR_CHECK(cudaMemcpy(host_outputImage, device_outputImage, sizeof(uchar3) * numPixels, cudaMemcpyDeviceToHost));
  }

  //free Memory
  CUDA_ERROR_CHECK(cudaFree(device_inputImage));
  CUDA_ERROR_CHECK(cudaFree(device_outputImage));

  printf("Size");

  return timeElapsedInMs;
}
