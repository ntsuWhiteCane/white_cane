#include <algorithm>
#include <eigen3\Eigen\Dense>
#include <iostream>
#include <vector>


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "initial_map.hpp"
#include "path.hpp"
#include "white_cane.hpp"

#define ADDTION_POINT_INTERVAL 30 
#define RESOLUTION 0.04

using namespace cv;
using namespace std;

int vertex_size = 0;
int test_p = 1;

bool simulation = true;
string map_path = "../map/lab_0623_1.pgm";
// string map_path = "../map/lin.png";
// string map_path = "../map/test_image.png";

//robot postition
Point2f robot_position;

// Point2f goal_position = Point2f(100, 100);
Point2f goal_position = Point2f(199, 200);


namespace myNameSpace{
	void myImshow(string windows_name, Mat img){
		if (simulation){
			namedWindow(windows_name, 0);
			imshow(windows_name, img);
		}
	}

	void myWaitKey(int time = 0){
		if (simulation){
			waitKey(time);
		}
	}
}

int main(int argc, char **argv){
	double start_time, end_time;
	robot_position = Point2f(399, 174);

	Mat map = imread(map_path);
	WhiteCane whiteCane(map);
	whiteCane.run();
	return 0;
}