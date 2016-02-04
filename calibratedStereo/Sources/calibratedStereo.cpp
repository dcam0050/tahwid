/*
 * calibratedStereo.cpp
 *
 *  Created on: 29 Jul 2015
 *      Author: root
 */

#include "calibratedStereo.hpp"

//performance 16fps on GTX675MX / i7
int main(int argc, char** argv)
{
	std::string imageInLeftPort;
	std::string imageInRightPort;
	std::string calibLeft;
	std::string calibStereoLeft;
	std::string calibRight;
	std::string map3D;
	std::string disp3D;
	bool disp3DFlag;
	std::string message;


	char mess[100];
	int numGPU = 0;

	yarp::os::Network yarp;

	if(argc < 8)
	{
		cout << "Not enough arguments. Must provide port name to the input and output ports" << endl;
		cout << "Exiting ..." << endl;
		return -1;
	}
	else
	{
		imageInLeftPort = argv[1];
		imageInRightPort = argv[2];
		calibLeft = argv[3];
		calibRight = argv[4];
		calibStereoLeft = argv[5];
		map3D = argv[6];
		disp3D = argv[7];
	}

	if (strcmp(disp3D.c_str(), "disparity") == 0)
	{
		disp3DFlag = 0; //output disparity map
	}
	else if (strcmp(disp3D.c_str(), "pointCloud") == 0)
	{
		disp3DFlag = 1; //output 3D point cloud
	}
	else
	{
		cout << "Invalid 3D output parameter. Please specify either 'disparity' or 'pointCloud'" << endl;
		return 0;
	}

	numGPU = cuda::getCudaEnabledDeviceCount();


	if (numGPU == 0)
	{
		cout << "No GPU found or library compiled without GPU support" << endl;
		cout << "Exiting ..." << endl;
		return 0;
	}

	for (int i = 0; i < numGPU; i++)
	{
		cuda::setDevice(i);
		cuda::DeviceInfo GPUlist;
		bool compatible = GPUlist.isCompatible();
		if (compatible == false)
		{
			cout << "Library not compiled with appropriate architecture" << endl;
			cout << "Exiting ..." << endl;
			return 0;
		}
	}

	cout << "Found " << numGPU << " CUDA enabled device/s" << endl;
	cv::cuda::Stream stream[5];

	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > imageInLeft;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > imageInRight;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > leftCalib;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > leftCalibStereoPort;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > rightCalib;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgbFloat> > stereoPortFloat;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > stereoPort;

	bool outOpen3;
	bool inOpenLeft = imageInLeft.open(imageInLeftPort.c_str());
	bool inOpenRight = imageInRight.open(imageInRightPort.c_str());
	bool outOpen1 = leftCalib.open(calibLeft.c_str());
	bool outOpen2 = rightCalib.open(calibRight.c_str());
	if(disp3DFlag) outOpen3 = stereoPortFloat.open(map3D.c_str());
	else outOpen3 = stereoPort.open(map3D.c_str());
	bool outOpen4 = leftCalibStereoPort.open(calibStereoLeft.c_str());

	if (!inOpenLeft | !inOpenRight | !outOpen1 | !outOpen2 | !outOpen3)
	{
		cout << "Could not open ports. Exiting" << endl;
		return -1;
	}

	double t = 0, time = 0, time2 = 0;
	int inCountLeft = 0;
	int inCountRight = 0;
	int outCount = 0;
	int step = 0;
	int count = 0;
	Mat captureFrame_cpuGBR, motion0, enhancedFeatures;
	cv::Ptr<cv::bioinspired::Retina> myRetinaLeft;
	cv::Ptr<cv::bioinspired::Retina> myRetinaRight;

#ifdef devDebug
	{
		yarp.connect("/icub/cam/left", "/imin/left", "udp+mjpeg+recv.bayer+method.nearest");
		yarp.connect("/icub/cam/right", "/imin/right", "udp+mjpeg+recv.bayer+method.nearest");
		//yarp.connect("/icub/cam/left", "/imin/left", "udp+mjpeg");
		//yarp.connect("/icub/cam/right", "/imin/right", "udp+mjpeg");
		//yarp.connect("/calibrated/left", "/imin/left");
		//yarp.connect("/calibrated/right", "/imin/right");

		//yarp.connect("/calib/left", "/showLeft");
		//yarp.connect("/calib/right", "/showRight");
		yarp.connect("/calib/right", "/showRight");
		yarp.connect("/calib/left", "/showLeft");
		yarp.connect("/calib/pointCloud", "/show3D");
		yarp.connect("/calib/stereoOut", "/showStereoLeft");
	}
#endif

	Mat intrinsicParamsLeft, distortionParamsLeft, intrinsicParamsRight, distortionParamsRight, stereoIntrinsicsRight, stereoIntrinsicsLeft,
		stereoDistortionRight, stereoDistortionLeft, cLeftMap1_1, cLeftMap1_2, cRightMap1_1, cRightMap1_2, cLeftMap2_1, cLeftMap2_2, cRightMap2_1, cRightMap2_2,
		cLeftCorrect, cRightCorrect, R, pano, stereoR, stereoT, stereoR_left, stQ,
		stereoR_right, stereoP_left, stereoP_right, stereoQ, left_cpuRGB1, left_cpuRGB2, right_cpuRGB1, right_cpuRGB2;

	cv::cuda::GpuMat gLeftCorrect, gLeftCorrect1, gRightCorrect, gRightCorrect1, gLeft, gRight, gLeftMap1_1, gLeftMap1_2, gLeftMap2_1,
					 gLeftMap2_2, gRightMap1_1, gRightMap1_2, gRightMap2_1, gRightMap2_2;

	cv::Rect roi1, roi2;

	//load individual left camera calibration parameters
	FileStorage camLeftParams = FileStorage();
	bool opened = camLeftParams.open("/home/icub/new_opencv/calibrationParams/left_camera_calib.yml", FileStorage::READ);
	if(opened)
	{
		camLeftParams["camera_matrix"] >> intrinsicParamsLeft;
		camLeftParams["distortion_coefficients"] >> distortionParamsLeft;
		camLeftParams.release();
	}
	else
	{
		cout << "left camera calibration file has not opened" <<  endl;
		return -1;
	}

	//load individual right camera calibration parameters
	FileStorage camRightParams = FileStorage();
	opened = camRightParams.open("/home/icub/new_opencv/calibrationParams/right_camera_calib.yml", FileStorage::READ);
	if(opened)
	{
		camRightParams["camera_matrix"] >> intrinsicParamsRight;
		camRightParams["distortion_coefficients"] >> distortionParamsRight;
		camRightParams.release();
	}
	else
	{
		cout << "right camera calibration failed to open" <<  endl;
		return -1;
	}

	//load stereo intrinsic calibration parameters
	FileStorage stereoIntrinsics = FileStorage();
	opened = stereoIntrinsics.open("/home/icub/new_opencv/calibrationParams/intrinsics_10deg.yml", FileStorage::READ);
	if(opened)
	{
		stereoIntrinsics["Mleft"] >> stereoIntrinsicsLeft;
		stereoIntrinsics["Mright"] >> stereoIntrinsicsRight;
		stereoIntrinsics["Dleft"] >> stereoDistortionLeft;
		stereoIntrinsics["Dright"] >> stereoDistortionRight;
		stereoIntrinsics.release();
	}
	else
	{
		cout << "stereo intrinsics file failed to open" <<  endl;
		return -1;
	}


	//load stereo extrinsic calibration parameters
	FileStorage stereoExtrinsics = FileStorage();
	opened = stereoExtrinsics.open("/home/icub/new_opencv/calibrationParams/extrinsics_10deg.yml", FileStorage::READ);
	if(opened)
	{
		stereoExtrinsics["R"] >> stereoR;
		stereoExtrinsics["T"] >> stereoT;
		stereoExtrinsics["Rleft"] >> stereoR_left;
		stereoExtrinsics["Pleft"] >> stereoP_left;
		stereoExtrinsics["Rright"] >> stereoR_right;
		stereoExtrinsics["Pright"] >> stereoP_right;
		stereoExtrinsics["Q"] >> stQ;
		stereoExtrinsics["ROI1"] >> roi1;
		stereoExtrinsics["ROI2"] >> roi2;
		stereoExtrinsics.release();
	}
	else
	{
		cout << "stereo extrinsics file failed to open" <<  endl;
		return -1;
	}

	stQ.convertTo(stereoQ, CV_32F);

	//acquire image to setup retina parameters
	bool setup = false;
	Size imageSize, inputSize;
	int border = 100;
	while (setup == false)
	{
		inCountLeft = imageInLeft.getInputCount();
		inCountRight = imageInRight.getInputCount();

		if (inCountLeft == 0 || inCountRight == 0)
		{
			cout << "Awaiting input images" << endl;
			pauseExec(100);
		}
		else
		{
			ImageOf<PixelRgb> *leftImage = imageInLeft.read();
			ImageOf<PixelRgb> *rightImage = imageInRight.read();
			if (leftImage != NULL && rightImage != NULL)
			{
				count = 0;
				step = leftImage->getRowSize() + leftImage->getPadding();
				Mat left_cpuRGB(leftImage->height(), leftImage->width(), CV_8UC3, leftImage->getRawImage(), step);
				Mat right_cpuRGB(rightImage->height(), rightImage->width(), CV_8UC3, rightImage->getRawImage(), step);
				inputSize = left_cpuRGB.size();
				Mat pano;

				//add borders to pad image in order to get better undistortion
				cv::copyMakeBorder(left_cpuRGB, left_cpuRGB, border, border, border, border, BORDER_CONSTANT, 0);
				cv::copyMakeBorder(right_cpuRGB, right_cpuRGB, border, border, border, border, BORDER_CONSTANT, 0);

				imageSize = left_cpuRGB.size();

				//step 1 warp map undistorting individual cameras
				initUndistortRectifyMap(intrinsicParamsLeft, distortionParamsLeft, R, intrinsicParamsLeft, imageSize, CV_32FC1, cLeftMap1_1, cLeftMap1_2);
				initUndistortRectifyMap(intrinsicParamsRight, distortionParamsRight, R, intrinsicParamsRight, imageSize, CV_32FC1, cRightMap1_1, cRightMap1_2);

				//step 2 warp map rectifying images
				initUndistortRectifyMap(stereoIntrinsicsLeft, stereoDistortionLeft, stereoR_left, stereoP_left, imageSize, CV_32FC1, cLeftMap2_1, cLeftMap2_2);
				initUndistortRectifyMap(stereoIntrinsicsRight, stereoDistortionRight, stereoR_right, stereoP_right, imageSize, CV_32FC1, cRightMap2_1, cRightMap2_2);

				//combine warp maps into a single warp map
				cv::remap(cLeftMap1_1, cLeftMap1_1, cLeftMap2_1, cLeftMap2_2, INTER_LINEAR);
				cv::remap(cLeftMap1_2, cLeftMap1_2, cLeftMap2_1, cLeftMap2_2, INTER_LINEAR);

				cv::remap(cRightMap1_1, cRightMap1_1, cRightMap2_1, cRightMap2_2, INTER_LINEAR);
				cv::remap(cRightMap1_2, cRightMap1_2, cRightMap2_1, cRightMap2_2, INTER_LINEAR);

				//retina process to hdr input images currently too slow to implement fast enough

				myRetinaLeft = cv::bioinspired::createRetina(inputSize);
				myRetinaLeft->setupOPLandIPLParvoChannel(true, true, 0.89f, 0.5f, 0.53f, 0.3f, 1.0f, 7.0f, 0.89f);
				myRetinaLeft->clearBuffers();

				myRetinaRight = cv::bioinspired::createRetina(inputSize);
				myRetinaRight->setupOPLandIPLParvoChannel(true, true, 0.89f, 0.5f, 0.53f, 0.3f, 1.0f, 7.0f, 0.89f);
				myRetinaRight->clearBuffers();

				setup = true;
			}
			else
			{
				count++;
				if (count == 50)
				{
					cout << "No input image detected" << endl;
					return 0;
				}
			}
		}
	}

	int numberOfDisparities = 128;

	Ptr<cuda::StereoBM> cudaBM;
	Ptr<cuda::DisparityBilateralFilter> cudaDBF;
	roi1.width = roi1.width - 20;
	roi2.width = roi2.width - 20;
	Rect roiCommon = roi1 & roi2;
	//Block Matching Settings
	cudaBM = cuda::createStereoBM();
	cudaBM->setROI1(roi1);
	cudaBM->setROI2(roi2);
	cudaBM->setPreFilterCap(31);
	cudaBM->setBlockSize(3);
	cudaBM->setMinDisparity(0);
	cudaBM->setNumDisparities(numberOfDisparities);
	cudaBM->setTextureThreshold(10);
	cudaBM->setUniquenessRatio(10);
	cudaBM->setSpeckleWindowSize(100);
	cudaBM->setSpeckleRange(32);
	cudaBM->setDisp12MaxDiff(1);

	cudaDBF = cuda::createDisparityBilateralFilter();
	cudaDBF->setRadius(5);
	cudaDBF->setNumIters(1);


	Mat cDisparityBM(imageSize, CV_8U), cDisparityBMoffset, cLeftCalib, cRightCalib, cDisparity(imageSize, CV_8U), cLeftCalibStereo;
	cuda::GpuMat gDisparityBM(imageSize, CV_8UC1), gLeftCorrectBM(imageSize, CV_8UC1), gRightCorrectBM(imageSize, CV_8UC1),
				 gDisparityLeftBM, gDisparityRightBM, gDisparityBMoffset,g3D, gLeftStereo, gStereoQ, gLeftDown, gRightDown, gMask;
	std::vector<cuda::GpuMat> xyz;

	gRightMap1_1.upload(cRightMap1_1);
	gRightMap1_2.upload(cRightMap1_2);

	gLeftMap1_1.upload(cLeftMap1_1);
	gLeftMap1_2.upload(cLeftMap1_2);
	gStereoQ.upload(stereoQ);

	Rect LeftROI, RightROI;
	Point leftPt, rightPt;
	Size leftSize, rightSize;
	double maxThresh = 2000, minThresh = 10;

	leftPt = Point(150, 80);
	leftSize = Size(690, 520);

	rightPt = Point(0, 35);
	rightSize = Size(690, 520);
	LeftROI = Rect(leftPt, leftSize);
	RightROI = Rect(rightPt, rightSize);

	while(true)
	{
		inCountLeft = imageInLeft.getInputCount();
		inCountRight = imageInRight.getInputCount();
		outCount = leftCalib.getOutputCount() + rightCalib.getOutputCount() + stereoPort.getOutputCount();
		if (inCountLeft == 0 || inCountRight == 0 || outCount == 0)
		{
			cout << "Awaiting input and output connections" << endl;
			pauseExec(100);
		}
		else
		{
			ImageOf<PixelRgb> *leftImage = imageInLeft.read();
			ImageOf<PixelRgb> *rightImage = imageInRight.read();
			if (leftImage != NULL && rightImage != NULL)
			{
					t = (double)getTickCount();
					count = 0;

					Mat cLeft_in(leftImage->height(), leftImage->width(), CV_8UC3, leftImage->getRawImage(), step);
					Mat cRight_in(rightImage->height(), rightImage->width(), CV_8UC3, rightImage->getRawImage(), step);

					//myRetinaLeft->applyFastToneMapping(cLeft_in, cLeft_in);
					//myRetinaRight->applyFastToneMapping(cRight_in, cRight_in);

					gLeft.upload(cLeft_in,stream[0]);
					gRight.upload(cRight_in, stream[1]);

					cuda::cvtColor(gLeft, gLeft, COLOR_RGB2BGR, -1, stream[0]);
					cuda::cvtColor(gRight, gRight, COLOR_RGB2BGR, -1, stream[1]);

					cuda::copyMakeBorder(gLeft, gLeft, border, border, border, border, BORDER_CONSTANT, 0, stream[0]);
					cuda::copyMakeBorder(gRight, gRight, border, border, border, border, BORDER_CONSTANT, 0, stream[1]);

					cuda::remap(gLeft, gLeftCorrect1, gLeftMap1_1, gLeftMap1_2, INTER_LINEAR, BORDER_CONSTANT, 0, stream[0]);
					cuda::remap(gRight, gRightCorrect1, gRightMap1_1, gRightMap1_2, INTER_LINEAR, BORDER_CONSTANT, 0, stream[1]);

					cuda::resize(gLeftCorrect1(roiCommon), gLeftStereo, inputSize,0,0,1,stream[2]);
					gLeftStereo.download(cLeftCalibStereo, stream[3]);

					cuda::resize(gLeftCorrect1(LeftROI), gLeftDown, inputSize, 0, 0, 1, stream[2]);
					gLeftDown.download(cLeftCalib, stream[3]);

					cuda::resize(gRightCorrect1(RightROI), gRightDown, inputSize, 0, 0, 1, stream[2]);
					gRightDown.download(cRightCalib, stream[3]);

					cv::cuda::cvtColor(gLeftCorrect1, gLeftCorrectBM, cv::COLOR_BGR2GRAY, 0, stream[0]);
					cv::cuda::cvtColor(gRightCorrect1, gRightCorrectBM, cv::COLOR_BGR2GRAY, 0, stream[1]);

					gLeftCorrectBM.convertTo(gDisparityLeftBM, CV_8U, stream[0]);
					gRightCorrectBM.convertTo(gDisparityRightBM, CV_8U, stream[1]);

					cudaBM->compute(gDisparityLeftBM, gDisparityRightBM, gDisparityBM, stream[3]);
					cudaDBF->apply(gDisparityBM, gLeftCorrect1, gDisparityBM, stream[3]);

					if (disp3DFlag)
					{
						cuda::add(gDisparityBM, 1, gDisparityBM);
						cuda::reprojectImageTo3D(gDisparityBM, g3D, stereoQ, 3);
						cuda::resize(g3D(roiCommon), g3D, inputSize);
						cuda::split(g3D, xyz);
						cuda::threshold(xyz[2], xyz[2], minThresh, 0, THRESH_TOZERO);
						cuda::threshold(xyz[2], xyz[2], maxThresh, 0, THRESH_TOZERO_INV);
						cuda::threshold(xyz[2], gMask, 0, 1, THRESH_BINARY);
						cuda::multiply(xyz[0], gMask, xyz[0]);
						cuda::multiply(xyz[1], gMask, xyz[1]);
						cuda::merge(xyz, g3D);
						g3D.download(cDisparityBM);
					}
					else
					{
						cuda::drawColorDisp(gDisparityBM, g3D, numberOfDisparities);
						cv::cuda::cvtColor(g3D, g3D, COLOR_RGBA2RGB);
						cuda::resize(g3D(roiCommon), g3D, inputSize);
						g3D.download(cDisparityBM);

					}
					//imshow("Disp", cDisparityBM);
					//waitKey(1);
					sprintf(mess, "Comp = %1.3f Trans = %1.3f FPS = %2.2f", time2, time, 1/(time+time2));
					cv::putText(cLeftCalib, mess, cv::Point(30, 30), FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 0, 200), 1);

					time2 = ((((double)getTickCount() - t) / getTickFrequency())*0.5) + (time2*0.5);

					if (disp3DFlag)
					{
						ImageOf<PixelRgb>& leftCalibOut = leftCalib.prepare();
						ImageOf<PixelRgb>& rightCalibOut = rightCalib.prepare();
						ImageOf<PixelRgb>& leftCalibStereoOut = leftCalibStereoPort.prepare();
						ImageOf<PixelRgbFloat>& stereoOutFloat = stereoPortFloat.prepare();

						CVtoYarp(cLeftCalib, leftCalibOut);
						CVtoYarp(cLeftCalibStereo, leftCalibStereoOut);
						CVtoYarp(cRightCalib, rightCalibOut);
						CVtoYarp(cDisparityBM, stereoOutFloat, false);

						leftCalib.write();
						leftCalibStereoPort.write();
						rightCalib.write();
						stereoPortFloat.write();
					}
					else
					{
						ImageOf<PixelRgb>& leftCalibOut = leftCalib.prepare();
						ImageOf<PixelRgb>& rightCalibOut = rightCalib.prepare();
						ImageOf<PixelRgb>& leftCalibStereoOut = leftCalibStereoPort.prepare();
						ImageOf<PixelRgb>& stereoOut = stereoPort.prepare();

						CVtoYarp(cLeftCalib, leftCalibOut);
						CVtoYarp(cLeftCalibStereo, leftCalibStereoOut);
						CVtoYarp(cRightCalib, rightCalibOut);
						CVtoYarp(cDisparityBM, stereoOut);

						leftCalib.write();
						leftCalibStereoPort.write();
						rightCalib.write();
						stereoPort.write();
					}

					time = (((((double)getTickCount() - t) / getTickFrequency())-time2)*0.5) + (time*0.5);
				}
		}
	}
}

void pauseExec (int sleepms)
{
	#ifdef linux
	{
		usleep(sleepms * 1000);   // usleep takes sleep time in us (1 millionth of a second)
	}
	#else
	{
		Sleep(sleepms);
	}
	#endif
}
