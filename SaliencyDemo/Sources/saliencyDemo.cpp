#include "saliencyDemo.hpp"

using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::sig::draw;
using namespace yarp::sig::file;
using namespace cv;
using namespace yarp::dev;
using namespace std;

#undef min
#undef max

int main(int argc, char** argv)
{
	std::string imageInPortName;
	std::string stereoInPortName;
	std::string magnoInPortName;
	std::string saliencyOutPortName;
	std::string gazeOutPortName;
	std::string facesInPortName;

	Network yarp;

	if(argc < 6)
	{
		std::cout << "Not enough arguments. Must provide port name to the input and output ports" << endl;
		std::cout << "Exiting ..." << endl;
		return -1;
	}
	else
	{
		imageInPortName = argv[1];
		stereoInPortName = argv[2];
		magnoInPortName = argv[3];
		facesInPortName = argv[4];
		saliencyOutPortName = argv[5];
		gazeOutPortName = argv[6];
	}

	int numGPU = cuda::getCudaEnabledDeviceCount();


	if (numGPU == 0)
	{
		std::cout << "No GPU found or library compiled without GPU support" << endl;
		std::cout << "Exiting ..." << endl;
		return 0;
	}

	for (int i = 0; i < numGPU; i++)
	{
		cuda::setDevice(i);
		cuda::DeviceInfo GPUlist;
		bool compatible = GPUlist.isCompatible();
		if (compatible == false)
		{
			std::cout << "Library not compiled with appropriate architecture" << endl;
			std::cout << "Exiting ..." << endl;
			return 0;
		}
	}

	std::cout << "Found " << numGPU << " CUDA enabled device/s" << endl;
	cv::cuda::Stream cuStream[5];


	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > imageInPort;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgbFloat> > stereoInPort;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelMono> > magnoInPort;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelMono> > facesInPort;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelFloat> > saliencyOutPort;
	yarp::os::Port gazeOutPort;

	bool inOpen = imageInPort.open(imageInPortName.c_str());
	//bool inOpen2 = stereoInPort.open(stereoInPortName.c_str());
	bool inOpen4 = facesInPort.open(facesInPortName.c_str());
	//bool inOpen3 = magnoInPort.open(magnoInPortName.c_str());
	bool outOpen = saliencyOutPort.open(saliencyOutPortName.c_str());
	bool outOpen2 = gazeOutPort.open(gazeOutPortName.c_str());

	if(!inOpen | !inOpen4 |!outOpen | !outOpen2)
	{
		std::cout << "Could not open ports. Exiting" << endl;
		return -1;
	}

#ifdef devDebug
{
	//yarp.connect("/icub/cam/left","/imin","udp+mjpeg+recv.bayer+method.nearest");
	//yarp.connect("/icub/cam/left","/imin","udp+mjpeg");
	//yarp.connect("/icubcam", "/imin");
	yarp.connect("/icub/cam/left","/inputCalibImage", "udp+mjpeg+recv.bayer+method.nearest");
	//yarp.connect("/stereo/out","/inputStereoMap");
	//yarp.connect("/magno/out", "/magnoInput");
	yarp.connect("/salientOut", "/salientFacesIn");
	yarp.connect("/saliencyOut","/showSaliency");
	yarp.connect("/gazeOut", "/iKinGazeCtrl/mono:i");
}
#endif


//using vision for stereo cameras:
// for all features except for depth, stitch the two images together to remove overlap and thus including peripheral vision.
// after stitching process through retina model and extract features
// for depth, process the overlapping section of the image only then after getting a saliency map, reincorporate this into
// an image whose dimensions are equal to the rest by setting the remaining pixels = 0
//instead of using optical flow for motion detection use magnocellular pathway implemented in retina module to extract motion saliency
//use of several gpus is done by getting a list of devices and setting which gpu does which operation:
//- requires either optimisation of event sequences where each gpu has a separate task to implement or striping of image
//separate tasks could possibly be more processing efficient due to the several 'independent' tasks that need to be carried out
//
	bool setup = false;
	int count = 0;
	Size imageSize;
	int border = 100;
	int inCount = 0, inCount1 = 0, inCount2 = 0, inCount3 = 0;
	int outCount = 0;
	int step = 0;
	int stepMagno = 0;
	int stepStereo = 0;
	int stepFaces = 0;
	double t = 0.0, time = 0.0;

	while(setup == false)
	{
		inCount = imageInPort.getInputCount();
		//inCount1 = stereoInPort.getInputCount();
		//inCount2 = magnoInPort.getInputCount();
		inCount2 = facesInPort.getInputCount();
		if (inCount == 0)
		{
			cout << "Awaiting input images" << endl;
			pauseExec(100);
		}
		else
		{
			ImageOf<PixelRgb> *image = imageInPort.read();
			//ImageOf<PixelRgbFloat> *stereo = stereoInPort.read();
			//ImageOf<PixelMono> *magno = magnoInPort.read();
			ImageOf<PixelMono> *faces = facesInPort.read();
			if (image != NULL & faces != NULL)
			{
				count = 0;
				step = image->getRowSize() + image->getPadding();
				//stepMagno = magno->getRowSize() + magno->getPadding();
				//stepStereo = stereo->getRowSize() + stereo->getPadding();
				stepFaces = faces->getRowSize() + faces->getPadding();

				Mat left_cpuRGB(image->height(), image->width(), CV_8UC3, image->getRawImage(), step);
				imageSize = left_cpuRGB.size();

				setup = true;
			}
		}
	}

	int numPix = imageSize.height*imageSize.width;
	char mess[100];
	GpuMat Intensity_past(imageSize, CV_32F);
	Intensity_past.setTo(Scalar::all(0.0));

	//prepare gabor filters
	int gaborWidth = 5;
	Size gaborSize = Size(gaborWidth, gaborWidth);
	Mat cgabor_0, cgabor_45, cgabor_90, cgabor_135;
	double sigma = 1, lambda = 2.0, gamma = 1.0, phase = 0;
	double Gstep = CV_PI / 4.0;
	double orien[4] = { (2 * Gstep), Gstep, 0, (3 * Gstep) };
	cgabor_0 =   getGaborKernel(gaborSize, sigma, orien[0], lambda, gamma, phase, CV_32F);
	cgabor_45 =  getGaborKernel(gaborSize, sigma, orien[1], lambda, gamma, phase, CV_32F);
	cgabor_90 =  getGaborKernel(gaborSize, sigma, orien[2], lambda, gamma, phase, CV_32F);
	cgabor_135 = getGaborKernel(gaborSize, sigma, orien[3], lambda, gamma, phase, CV_32F);

	GpuMat Gabor0_0, Gabor0_90, Gabor0_45, Gabor0_135;

	Ptr<Filter> gaborFilt0 =   cuda::createLinearFilter(CV_32F, CV_32F, cgabor_0,   Point(-1, -1), BORDER_CONSTANT, 0);
	Ptr<Filter> gaborFilt90 =  cuda::createLinearFilter(CV_32F, CV_32F, cgabor_90,  Point(-1, -1), BORDER_CONSTANT, 0);
	Ptr<Filter> gaborFilt45 =  cuda::createLinearFilter(CV_32F, CV_32F, cgabor_45,  Point(-1, -1), BORDER_CONSTANT, 0);
	Ptr<Filter> gaborFilt135 = cuda::createLinearFilter(CV_32F, CV_32F, cgabor_135, Point(-1, -1), BORDER_CONSTANT, 0);

	//prepare pyramid sizes array
	const int numPyrLevels = 8;
	int numFMaps = 6;
	int conspMapLevel = 4;
	vector<int> centreVec;
	centreVec.push_back(2);
	centreVec.push_back(3);
	centreVec.push_back(4);
	vector<int> surroundOffsetVec;
	surroundOffsetVec.push_back(3);
	surroundOffsetVec.push_back(4);

	vector<GpuMat> gMagnoPYR, gMagnoFMaps, gIntensityPYR, gIntensityFMaps, RGantPYR, RGantFMaps, BYantPYR, BYantFMaps, Gabor0PYR, Gabor90PYR, Gabor45PYR, Gabor135PYR, Gabor0FMaps, Gabor90FMaps, Gabor45FMaps, Gabor135FMaps;
	vector<Mat> intensity;
	Mat intPast, IOR;
	GpuMat gTempY, gTempX, gRG;
	Size pyrSizes[numPyrLevels+1];
	Rect pyrRect[numPyrLevels+1];
	Point origin = Point(0, 0);
	int tempH, tempW;

	Intensity_past.download(intPast);


	for (int i = 0; i <= numPyrLevels; i++)
	{
		tempH = ceil(imageSize.height / pow(2,i));
		tempW = ceil(imageSize.width / pow(2, i));
		pyrSizes[i] = Size(tempW, tempH);
		pyrRect[i] = Rect(origin, pyrSizes[i]);
		gIntensityPYR.push_back(Intensity_past.clone());
		//gMagnoPYR.push_back(Intensity_past.clone());
		RGantPYR.push_back(Intensity_past.clone());
		BYantPYR.push_back(Intensity_past.clone());
		Gabor0PYR.push_back(Intensity_past.clone());
		Gabor90PYR.push_back(Intensity_past.clone());
		Gabor45PYR.push_back(Intensity_past.clone());
		Gabor135PYR.push_back(Intensity_past.clone());
		if (i < numFMaps)
		{
			gIntensityFMaps.push_back(Intensity_past.clone());
			//gMagnoFMaps.push_back(Intensity_past.clone());
			RGantFMaps.push_back(Intensity_past.clone());
			BYantFMaps.push_back(Intensity_past.clone());
			Gabor0FMaps.push_back(Intensity_past.clone());
			Gabor90FMaps.push_back(Intensity_past.clone());
			Gabor45FMaps.push_back(Intensity_past.clone());
			Gabor135FMaps.push_back(Intensity_past.clone());
		}
		intensity.push_back(intPast);
	}

	GpuMat gIntensityConspMap(pyrSizes[conspMapLevel], CV_32F);
	//GpuMat gMagnoConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat RGantConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat BYantConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat Gabor0ConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat Gabor90ConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat Gabor45ConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat Gabor135ConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat OrientationConspMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat IORMap(pyrSizes[conspMapLevel], CV_32F);
	GpuMat gMask(Size(3, numPix), CV_8UC1);

	IORMap.setTo(Scalar::all(0.0));
	//initialise maximum filtering
	//Ptr<Filter> maxFilt = cuda::createBoxMaxFilter(CV_8UC1, Size(5, 5), Point(-1, -1), BORDER_CONSTANT, 0);
	Mat kernel, cSaliencyInt;
	kernel = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	Ptr<Filter> maxFilt = cuda::createMorphologyFilter(MORPH_DILATE, CV_8UC1, kernel, Point(-1, -1), 1);

	double avgFps, cumuTime = 0, avgTime, minSalVal, maxSalVal;
	Point minSalPoint, maxSalPoint;
	Mat m1, m2, m3, m0, imageFloat, cSaliency, magnoFloat, facesFloat;
	GpuMat gTemp3, gMaxRGB, gMinRG, gMagnoDisparity, gRightInt, gFaces, gFacesConspMap, gMagno3D, gLeftInt, gLeftROI, gMagnoROI, gMagnoLeftCorrect1, gMagnoRightCorrect1, gMagnoRightInt, gMagnoLeftInt,  gRight, gR, gMagnoLeft, gMagnoRight, gMagno0, gG, gB, gIntensityCurr, gIntensity0, gLeft, pano, RGant0, temp, BYant0, Flicker0, gTemp, gTemp2, gTemp4, gTemp5, gTemp6, gTemp7, gTemp8, gSaliency, ColorConspMap;
	vector<GpuMat> bgr, rg;
	int countFrames = 0, countGaze = 0;
	Scalar arrMean, arrStddev;
	int lenX = pow(2,conspMapLevel);
	int lenY = lenX;
	int salX = 0, salY = 0;
	int cumulativeX = 0, cumulativeY = 0;
	Rect salientRegion, saliencyMapRegion;
	Scalar meanPos;
	double decayFactor = 0.95;
	int gazePeriod = 6;
	int midX = floor(gazePeriod/2);
	std::vector<int> vecX(gazePeriod);
	std::vector<int> vecY(gazePeriod);
	std::vector<int> sortX(gazePeriod);
	vector<int>::iterator medianIdx;
	int medianX;
	int salIdx;
	//std::deque<Point> saccadePts;
	//saccadePts.push_back(Point(0,0));
	//saccadePts.push_back(Point(0,0));
	//saccadePts.push_back(Point(0,0));

	Point2f avgPoint;

	while(true)
	{
		inCount = imageInPort.getInputCount();
		//inCount1 = stereoInPort.getInputCount();
		//inCount2 = magnoInPort.getInputCount();
		inCount3 = facesInPort.getInputCount();
		outCount = saliencyOutPort.getOutputCount();

		if(inCount == 0 || inCount3 == 0 || outCount == 0)
		{
			std::cout << "Awaiting input and output connections" << endl;
			pauseExec(100);
		}
		else
		{
			ImageOf<PixelRgb> *imageIn = imageInPort.read();
			//ImageOf<PixelRgbFloat> *stereoIn = stereoInPort.read();
			//ImageOf<PixelMono> *magnoIn = magnoInPort.read();
			ImageOf<PixelMono> *facesIn = facesInPort.read();
			if ((imageIn != NULL) & (facesIn != NULL))
			{
				t = (double)getTickCount();
				Mat inImage(imageIn->height(), imageIn->width(), CV_8UC3, imageIn->getRawImage(), step);
				//Mat stereoImage(stereoIn->height(), stereoIn->width(), CV_32FC3, stereoIn->getRawImage(), stepStereo);
				//Mat magnoImage(magnoIn->height(), magnoIn->width(), CV_8UC1, magnoIn->getRawImage(), stepMagno);
				Mat facesImage(facesIn->height(), facesIn->width(), CV_8UC1, facesIn->getRawImage(), stepFaces);

				cv::cvtColor(inImage, inImage, COLOR_RGB2BGR, -1);
				//magnoImage.convertTo(magnoFloat, CV_32FC1);
				//magnoFloat /= 255;

				inImage.convertTo(imageFloat, CV_32FC3);
				imageFloat /= 255;

				facesImage.convertTo(facesFloat, CV_32FC1);
				facesFloat /= 255;

				gLeft.upload(imageFloat);
				//gMagno0.upload(magnoFloat);
				gFaces.upload(facesFloat);

				//------------ start of saliency---------------------//

				//create intensity map
				cuda::split(gLeft, bgr, cuStream[0]);

				cuda::add(bgr[1], bgr[0], gTemp2, noArray(), -1, cuStream[0]);
				cuda::addWeighted(gTemp2, 0.33333, bgr[2], 0.33333, 0, gIntensity0, -1, cuStream[0]);
				cuda::threshold(gIntensity0, gMask, 0.1, 1, THRESH_BINARY, cuStream[0]);
				cudaConspMap(&gIntensity0, &gIntensityPYR, &gIntensityFMaps, &gIntensityConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);

				//creating colour antagonist maps
				cuda::max(bgr[0], bgr[1], gTemp2, cuStream[0]);
				cuda::max(gTemp2, bgr[2], gTemp3, cuStream[0]);
				cuda::multiply(gTemp3, gMask, gMaxRGB, 1.0, -1, cuStream[0]);

				cuda::min(bgr[2], bgr[1], gMinRG, cuStream[0]);

				cuda::subtract(bgr[2], bgr[1], RGant0, noArray(), -1, cuStream[0]);
				cuda::divide(RGant0, gMaxRGB, RGant0, 1, -1, cuStream[0]);
				cudaConspMap(&RGant0, &RGantPYR, &RGantFMaps, &RGantConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);
				cuda::add(gIntensityConspMap, RGantConspMap, gSaliency);

				cuda::subtract(bgr[0], gMinRG, BYant0, noArray(), -1, cuStream[0]);
				cuda::divide(BYant0, gMaxRGB, BYant0, 1, -1, cuStream[0]);
				cudaConspMap(&BYant0, &BYantPYR, &BYantFMaps, &BYantConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);
				cuda::add(gSaliency, BYantConspMap, gSaliency);

				cuda::resize(gFaces, gFacesConspMap, pyrSizes[conspMapLevel], 0.0, 0.0, 1.0, cuStream[1]);

				//--------------------------------------------------------//

			//flicker map (use retina magno channel)
				//cudaConspMap(&gMagno0, &gMagnoPYR, &gMagnoFMaps, &gMagnoConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);

			//gabor filtering on intensity map
				gaborFilt0->apply(gIntensity0, Gabor0_0, cuStream[0]);
				cudaConspMap(&Gabor0_0, &Gabor0PYR, &Gabor0FMaps, &Gabor0ConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);
				gaborFilt90->apply(gIntensity0, Gabor0_90, cuStream[0]);
				cudaConspMap(&Gabor0_90, &Gabor90PYR, &Gabor90FMaps, &Gabor90ConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);
				gaborFilt45->apply(gIntensity0, Gabor0_45, cuStream[0]);
				cudaConspMap(&Gabor0_45, &Gabor45PYR, &Gabor45FMaps, &Gabor45ConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);
				gaborFilt135->apply(gIntensity0, Gabor0_135, cuStream[0]);
				cudaConspMap(&Gabor0_135, &Gabor135PYR, &Gabor135FMaps, &Gabor135ConspMap, numPyrLevels, centreVec, surroundOffsetVec, conspMapLevel, pyrSizes, maxFilt, cuStream[1]);

				OrientationConspMap.setTo(Scalar::all(0.0));
				normImage(&Gabor0ConspMap, maxFilt, &Gabor0ConspMap, pyrSizes[conspMapLevel], cuStream[1]);
				cuda::add(OrientationConspMap, Gabor0ConspMap, OrientationConspMap);

				normImage(&Gabor90ConspMap, maxFilt, &Gabor90ConspMap, pyrSizes[conspMapLevel], cuStream[1]);
				cuda::add(OrientationConspMap, Gabor90ConspMap, OrientationConspMap);

				normImage(&Gabor45ConspMap, maxFilt, &Gabor45ConspMap, pyrSizes[conspMapLevel], cuStream[1]);
				cuda::add(OrientationConspMap, Gabor45ConspMap, OrientationConspMap);

				normImage(&Gabor135ConspMap, maxFilt, &Gabor135ConspMap, pyrSizes[conspMapLevel], cuStream[1]);
				cuda::add(OrientationConspMap, Gabor135ConspMap, OrientationConspMap);

				cuda::addWeighted(gSaliency, 0.333, OrientationConspMap, 0.333, 0, gSaliency, -1 ,cuStream[1]);
				//cuda::addWeighted(gSaliency, 0.75 , gMagnoConspMap, 0.25, 0, gSaliency, -1, cuStream[1]);
				cuda::addWeighted(gSaliency, 0.5, gFacesConspMap, 0.5, 0, gSaliency, -1, cuStream[1]);
				cuda::multiply(gSaliency,255, gSaliency, 1.0, -1, cuStream[1]);
				cuda::subtract(gSaliency, IORMap, gSaliency, noArray(), CV_32FC1, cuStream[1]);
				cuda::minMaxLoc(gSaliency, &minSalVal, &maxSalVal, &minSalPoint, &maxSalPoint, noArray());				//--------------------------------------------------//

				gSaliency.download(cSaliency);

				salX = maxSalPoint.x;
				salY = maxSalPoint.y;
				if(salX == 0) salX = 1;
				if(salY == 0) salY = 1;
				saliencyMapRegion = Rect(salX, salY, 1, 1);

				salX = floor(salX*pow(2,conspMapLevel));
				salY = floor(salY*pow(2,conspMapLevel));

				//cumulativeX += salX;
				//cumulativeY += salY;
				vecX[countGaze] = salX;
				vecY[countGaze] = salY;

				salientRegion = Rect(salX-ceil(lenX/2), salY-ceil(lenY/2), lenX, lenY);

				rectangle(inImage, salientRegion, Scalar(0, 0, 200), 3, 8, 0);
				imshow("salientAttention", inImage);
				waitKey(1);

				countGaze++;
				if(countGaze == gazePeriod)
				{
					countGaze = 0;
					sortX = vecX;
					nth_element(sortX.begin(), sortX.begin()+midX, sortX.end());
					medianX = sortX[midX];
					medianIdx = std::find(vecX.begin(), vecX.end(), medianX);
					salIdx = medianIdx - vecX.begin();
					//vecX.begin()

					Bottle gaze;
					gaze.clear();
					gaze.addString("left");
					gaze.addDouble(vecX[salIdx]);
					gaze.addDouble(vecY[salIdx]);
					gaze.addDouble(1.0);
					gazeOutPort.write(gaze);
					cout << "gaze written" << endl;
					cumulativeX = 0;
					cumulativeY = 0;
					cuda::multiply(IORMap, decayFactor, IORMap, 1.0, CV_32FC1, cuStream[1]);
					cuda::add(IORMap(saliencyMapRegion), 500, IORMap(saliencyMapRegion), noArray(), CV_32FC1, cuStream[1]);
				}

				time = (1000/(getTickFrequency() / ((double)getTickCount() - t)));
				cumuTime += time;
				countFrames++;
				avgTime = cumuTime / countFrames;
				avgFps = 1000 / avgTime;
				if (countFrames == 50)
				{
					cumuTime = 0;
					countFrames = 0;
				}
				//gDisparityBM.download(cDisparity);
				//g3D.download(cDisparity);
				//imshow("Disparity", cDisparity);

				//gMagnoDisparity.download(cMagnoDisparity);
				//gMagno3D.download(cMagnoDisparity);
				//imshow("Magno Disparity", cMagnoDisparity);
				//waitKey(1);

				//gIntensityConspMap.download(cSaliency, cuStream[1]);
				std::sprintf(mess, "Avg Time= %3.4f, Avg FPS = %3.2f", avgTime, avgFps);
				//putText(cSaliency, mess, cvPoint(30, 30), FONT_HERSHEY_COMPLEX, 0.2, cvScalar(200, 0, 0), 1, CV_AA);
				std::cout << mess << endl;
				//for (int i = 0; i < numFMaps; i++)
				//{
				//	gIntensityPYR[i].download(intensity[i]);
				//	if (i == 0) putText(intensity[i], mess, cvPoint(30, 30), FONT_HERSHEY_COMPLEX, 0.5, cvScalar(200, 0, 0), 1, CV_AA);
				//	sprintf(win, "window %d", i);
				//	namedWindow(win, WINDOW_NORMAL);
					//imshow(win, intensity[i]);
				//}

				//cv::namedWindow("Saliency Map", WINDOW_NORMAL);
				//cv::imshow("Saliency Map", cSaliency);

				//cv::waitKey(1);
				//cSaliency.convertTo(cSaliencyInt, CV_8UC1, 1, 0);
				yarp::sig::ImageOf<yarp::sig::PixelFloat>& saliencyYarpOut = saliencyOutPort.prepare();

				CVtoYarp(cSaliency, saliencyYarpOut);
				saliencyOutPort.write();
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

void cudaConspMap(cuda::GpuMat *gBaseFeatureMap, vector<cuda::GpuMat> *gFeaturePyramid, vector<cuda::GpuMat> *gFeatureMapsArray, GpuMat *gConspicuityMap, int numPyrLevels, vector<int> centreVec, vector<int> surroundOffsetVec, int conspMapLevel, Size pyrSizes[], Ptr<Filter> maxFilt, cuda::Stream cuStream)
{
	GpuMat gTemp, gTemp2, gTemp5;

	gConspicuityMap->setTo(Scalar::all(0.0));

	//check lenght of gFeaturePyramids and gFeatureMapsArray and data types, resolution of gBaseFeatureMap and data type, resolution of gConspicuityMap and data type,

	//future improvement make a conspicuity class containing all initialised settings and storing all the data required
	gFeaturePyramid->operator[](0) = *gBaseFeatureMap;
	for (int i = 0; i < numPyrLevels; i++)
	{
		cuda::pyrDown(gFeaturePyramid->operator[](i), gFeaturePyramid->operator[](i + 1), cuStream);
	}

	//the next step computes centre surround differences
	//feature map for arbitrary centre scale q and arbitrary surround scale s is
	//p(q,s) = |p(q) - p(s) linearly interpolated to level q
	//for conspicuity maps instead of taking this centre surround using one centre,
	//it is carried out for a range of centres with a range of surrounds for each centre

	int currCentre, currSurr, ind;
	for (int i = 0; i < centreVec.capacity(); i++)
	{
		currCentre = centreVec[i];
		for (int j = 0; j < surroundOffsetVec.capacity(); j++)
		{
			ind = i*surroundOffsetVec.capacity() + j;
			currSurr = currCentre + surroundOffsetVec[j];
			cuda::resize(gFeaturePyramid->operator[](currSurr), gTemp, pyrSizes[currCentre], 0.0, 0.0, 1, cuStream);
			cuda::absdiff(gFeaturePyramid->operator[](currCentre), gTemp, gFeatureMapsArray->operator[](ind), cuStream);
			cuda::resize(gFeatureMapsArray->operator[](ind), gTemp2, pyrSizes[conspMapLevel], 0.0, 0.0, 1, cuStream);
			normImage(&gTemp2, maxFilt, &gTemp5, pyrSizes[conspMapLevel], cuStream);
			cuda::add(*gConspicuityMap, gTemp5, *gConspicuityMap, noArray(), -1, cuStream);
		}
	}
}


void normImage(GpuMat* input, Ptr<Filter> maxFilt, GpuMat* output, Size pyrSize, cuda::Stream cuStream, bool choice)
{
	GpuMat gTemp3, gTemp4, gTemp5, gminmax, factor, gmean, diff;
	double maxVal, minVal;

	cuda::minMax(*input, &maxVal, &minVal, noArray(), gminmax);
	//cuda::findMinMax(*input, gminmax, noArray(), cuStream); //find minimum and maximum inimage
	cuda::subtract(gminmax(Rect(1, 0, 1, 1)), gminmax(Rect(0, 0, 1, 1)), factor, noArray(), -1, cuStream); //subtract min and max to get factor
	cuda::copyMakeBorder(factor, factor, 0, pyrSize.height - 1, pyrSize.width - 1, 0, BORDER_REPLICATE, 0, cuStream);

	if (choice == true)
	{
		Scalar arrMean, arrStddev;
		//option 1
		cuda::multiply(*input, factor, gTemp4, 255, -1, cuStream); //output normalised image data type float in gTemp4 range 0-255
		gTemp4.convertTo(gTemp3, CV_8UC1, cuStream); //convert to 8UC1 for max filtering gTemp3 conatins proper uint image//90fps here
		maxFilt->apply(gTemp3, gTemp5, cuStream); //70fps here
		cuda::meanStdDev(gTemp5, arrMean, arrStddev); //40 fps here
		double diff = pow(((255-arrMean.val[0])/255), 2)/255; //40fps
		cuda::multiply(gTemp4, diff, *output, 1.0, CV_32F, cuStream); //36fps
	}
	else
	{
		//option 2
		cuda::multiply(*input, factor, gTemp4, 1.0, -1, cuStream); //output normalised image data type float in gTemp4 range 0-255
		cuda::reduce(gTemp4, gmean, 0, REDUCE_MAX, -1, cuStream);
		cuda::reduce(gmean, gmean, 1, REDUCE_AVG, -1, cuStream);
		cuda::subtract(1.0, gmean, diff, noArray(), -1, cuStream);
		cuda::pow(diff, 2, diff, cuStream);
		cuda::copyMakeBorder(diff, diff, 0, pyrSize.height - 1, pyrSize.width - 1, 0, BORDER_REPLICATE, 0, cuStream);
		cuda::multiply(gTemp4, diff, *output, 1.0, -1, cuStream);
	}
}
