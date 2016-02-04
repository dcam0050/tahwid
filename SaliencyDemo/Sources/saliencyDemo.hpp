#include <stdio.h>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <sig/all.h>
#include <os/all.h>
#include <dev/all.h>
#include <ctime>
#include <numeric>
#include <functional>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/bioinspired.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <queue>
#include <algorithm>


#define devDebug

#ifdef linux
#include <unistd.h>
#else
#include <windows.h>
#endif

using namespace yarp::os;
using namespace yarp::sig;
using namespace cv;
using namespace std;
using namespace cuda;

void CVtoYarp(cv::Mat MatImage, ImageOf<PixelRgb> & yarpImage);
void CVtoYarp(cv::Mat MatImage, ImageOf<PixelRgbFloat> & yarpImage, bool flip = true);
void CVtoYarp(cv::Mat MatImage, yarp::sig::ImageOf<PixelMono> & yarpImage);
void CVtoYarp(cv::Mat MatImage, yarp::sig::ImageOf<PixelFloat> & yarpImage);
void pauseExec(int sleepms);
void cudaConspMap(cuda::GpuMat *gBaseFeatureMap, vector<cuda::GpuMat> *gFeaturePyramid, vector<cuda::GpuMat> *gFeatureMapsArray, GpuMat *gConspicuityMap, int numPyrLevels, vector<int> centreVec, vector<int> surroundOffsetVec, int conspMapLevel, Size pyrSizes[], Ptr<Filter> maxFilt, cuda::Stream cuStream);
void normImage(GpuMat* input, Ptr<Filter> maxFilt, GpuMat* output, Size pyrSize, cuda::Stream cuStream, bool choice = false);
