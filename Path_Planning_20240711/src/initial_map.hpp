#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#ifndef INITIAL_MAP_H
#define INITIAL_MAP_H
class InitialMap{
	public:
		InitialMap(cv::Mat &src, int addition_point_interval = 25,int max_corners = 200);
		~InitialMap();
		void cornerFun();
		void mapPreprocessing();
		void addtionPoint(cv::Mat &outputImage);
		std::vector<cv::Point2f> getCorner();
		std::vector<cv::Point2f> getAdditionPoint();
		cv::Mat getMap();
		cv::Mat getRGBMap();
		void setGrayMap(cv::Mat &inputImg);
		void setRGBMap(cv::Mat &inputImg);
		void grayMap2RGBMap();
		
	protected:
		std::vector<cv::Point2f> points;
		cv::Mat img;
		cv::Mat img_rgb;
	private:	
		int maxCorners = 1; // maxCorner is 200
		std::vector<cv::Point2f> corners;
		double qualityLevel = 0.01;
		double minDistance = 5;
		int blockSize = 3;
		int gradientSize = 3;
		bool useHarrisDetector = false;
		double k = 0.04;
		int addition_point_interval;
		
};
#endif