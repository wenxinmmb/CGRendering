#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace cv;

//defined in imageFilter.cu
float filterImage(uchar3 *host_inputImage, uchar3 *host_outputImage, int rows, int cols, int filterNumber);

int main( int argc, char** argv )
{
 char* imageName = argv[1];
 char *filterArg = argv[2];

 int filterNumber = filterArg[0] - '0';

 std::string filterName;

 switch(filterNumber){
   case 1: filterName = "channel_swap";
    break;
   case 2: filterName = "blur";
    break;
   case 3: filterName = "flip";
    break;
   case 4: filterName = "creative";
    break;
   default:
    printf( " Invalid Filter Argument \n " );
    return -1;
 }

 printf( "Applying Filter: %s \n" , filterName.c_str());

 //Load Input Image in BGR Format
 Mat inputImage;
 inputImage = imread( imageName, CV_LOAD_IMAGE_COLOR); // loads image in bgr format

 if( argc != 3 || !inputImage.data )
 {
   printf( " No image data \n " );
   return -1;
 }

//Allocate Memory for output image on host
Mat outputImage;
outputImage.create(inputImage.rows, inputImage.cols, CV_8UC3);

//Create pointers to the host input and output images, to be passed in to the filter image function
uchar3 *host_inputImage;
uchar3 *host_outputImage;

host_inputImage  = (uchar3 *)inputImage.ptr<unsigned char>(0);
host_outputImage = (uchar3 *)outputImage.ptr<unsigned char>(0);

//Call Filter Image Function (defined in imageFilter.cu)
//This should produce a filtered version of host_inputImage, and store it in host_outputImage
float timeElapsedInMs = filterImage(host_inputImage,host_outputImage,inputImage.rows,inputImage.cols, filterNumber);

printf("time elapsed in milliseconds for kernel call:%f\n", timeElapsedInMs);

//write the output image
imwrite( filterName + ".jpg", outputImage);

//- Only works if you're running some form of X-Forwarding (e.g. using FarmVNC and MobaXterm) - see assignment for details
// If you are, you can uncomment these lines!!
//Display the input image
//namedWindow(imageName, CV_WINDOW_AUTOSIZE );
//imshow(imageName, inputImage );

//- Only works if you're running some form of X-Forwarding (e.g. using FarmVNC and MobaXterm) - see assignment for details
// If you are, you can uncomment these lines!!
//Display the output image
//namedWindow("output image", CV_WINDOW_AUTOSIZE );
//imshow("output image", outputImage );
//waitKey(0);

 return 0;
}
