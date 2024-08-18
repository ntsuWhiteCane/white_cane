#include <vector>
#include "opencv2/core.hpp"

#ifndef MAIN_H
#define MAIN_H

#define ADDTION_POINT_INTERVAL 30 
#define RESOLUTION 0.04

extern int test_p;
extern bool simulation;
extern std::string map_path;
//robot postition
extern cv::Point2f robot_position;
extern cv::Point2f goal_position;


namespace myNameSpace{
	void myImshow(std::string windows_name, cv::Mat img);

	void myWaitKey(int time = 0);
}
#endif