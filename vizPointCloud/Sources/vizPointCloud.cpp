/*
 * calibratedStereo.cpp
 *
 *  Created on: 29 Jul 2015
 *      Author: root
 */

#include "vizPointCloud.hpp"

//performance 16fps on GTX675MX / i7
int main(int argc, char** argv){
	std::string cloudInPortName;
	std::string imageInPortName;
	cv::viz::Viz3d env3D("Environment");

	yarp::os::Network yarp;

	if(argc < 3)
	{
		cout << "Not enough arguments. Must provide port name to the input" << endl;
		cout << "Exiting ..." << endl;
		return -1;
	}
	else
	{
		cloudInPortName = argv[1];
		imageInPortName = argv[2];
	}

	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgbFloat> > cloudInPort;
	yarp::os::BufferedPort<yarp::sig::ImageOf<yarp::sig::PixelRgb> > imageInPort;

	bool inOpen1 = cloudInPort.open(cloudInPortName.c_str());
	bool inOpen2 = imageInPort.open(imageInPortName.c_str());

	if (!inOpen1 | !inOpen2)
	{
		cout << "Could not open ports. Exiting" << endl;
		return -1;
	}
	int inCount;

#ifdef devDebug
	yarp.connect("/calib/pointCloud", "/in3D");
	yarp.connect("/calib/stereoOut", "/inImage");
#endif

	env3D.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	env3D.spinOnce(1, true);

	int stepInt, stepFloat;
	while(!env3D.wasStopped())
	//while(true)
	{
		inCount = cloudInPort.getInputCount() + imageInPort.getInputCount();
		if (inCount < 2)
		{
			cout << "Awaiting input and output connections" << endl;
			pauseExec(100);
		}
		else
		{
			ImageOf<PixelRgbFloat> *pointCloud = cloudInPort.read();
			ImageOf<PixelRgb> *yimage = imageInPort.read();

			if (yimage!= NULL && pointCloud != NULL)
			{

				stepFloat = pointCloud->getRowSize() + pointCloud->getPadding();
				Mat points(pointCloud->height(), pointCloud->width(), CV_32FC3, pointCloud->getRawImage(), stepFloat);

				stepInt = yimage->getRowSize() + yimage->getPadding();
				Mat image(yimage->height(), yimage->width(), CV_8UC3, yimage->getRawImage(), stepInt);
				cv::cvtColor(image, image, COLOR_RGB2BGR);
				cv::viz::WCloud ptCloud(points, image);
				env3D.showWidget("cloud", ptCloud);
				//cv::namedWindow("points", WINDOW_NORMAL);
				//imshow("points", points);
				//waitKey(1);
			}
		}
		env3D.spinOnce(1, true);
	}
	return 0;
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
