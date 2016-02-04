/*
 * calibratedStereo.hpp
 *
 *  Created on: 29 Jul 2015
 *      Author: root
 */

#ifndef SOURCES_CALIBRATEDSTEREO_HPP_
#define SOURCES_CALIBRATEDSTEREO_HPP_

#include <stdio.h>
#include <sig/all.h>
#include <os/all.h>
#include <dev/all.h>
#include <ctime>
#include <opencv2/stitching.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/ocl_genbase.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/bioinspired.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <iostream>
#include <math.h>
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
using namespace cv::cuda;
using namespace cv::ocl;

void CVtoYarp(cv::Mat MatImage, yarp::sig::ImageOf<PixelRgb> & yarpImage);
void CVtoYarp(cv::Mat MatImage, yarp::sig::ImageOf<PixelRgbFloat> & yarpImage, bool flip = true);
void pauseExec(int sleepms);



#endif /* SOURCES_CALIBRATEDSTEREO_HPP_ */
