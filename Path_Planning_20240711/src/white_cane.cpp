#include <algorithm>
#include <eigen3\Eigen\Dense>
#include <iostream>
#include <vector>


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "main.hpp"
#include "initial_map.hpp"
#include "path.hpp"
#include "white_cane.hpp"

using namespace cv;
using namespace std;

WhiteCane::WhiteCane(Mat map){
    int inputImg_channels = map.channels();
    Mat voronoi_img;
	Mat obstacle_img;
    
    if(inputImg_channels == 1){
        this->img = map;
        cvtColor(map, this->img_rgb, COLOR_GRAY2BGR);
    }
    else if(inputImg_channels == 3){
        cvtColor(map, this->img, COLOR_BGR2GRAY);
        this->img_rgb = map;
    }
    else if(inputImg_channels == 4){
        cvtColor(map, this->img, COLOR_BGRA2GRAY);
        cvtColor(map, this->img_rgb, COLOR_BGRA2BGR);
    }
    else{
		cout << "Input image channel number is not valid." << endl;
		cout << "The number of channels in the input image can only be 1 (Gray Scale), 3 (RGB), or 4 (RGBA), but this image has " << inputImg_channels << " channels." << endl;
	}

	this->pathPlanning = new PathPlanning(this->img);
	pathPlanning->mapPreprocessing();
	pathPlanning->grayMap2RGBMap();
	pathPlanning->addtionPoint(obstacle_img);
	pathPlanning->voronoi(voronoi_img, this->full_vex_edge);

	myNameSpace::myImshow("Original Map", map);
	myNameSpace::myImshow("Preprocessing Map", this->img_rgb);
	myNameSpace::myImshow("Voronoi", voronoi_img);

}

WhiteCane::~WhiteCane(){
	delete this->pathPlanning;
}

void WhiteCane::run(){

	int num_cnt = 1, num_edge_cnt = 1;
	if (mix_vex_edge[0].vex_size == 0) {
		// do not detect obstacle at beginning
		obs_px = 3;
		obs_py = 3;
		for (int i = 0; i < full_vex_edge[0].vex_size; i++) {

			mix_vex_edge[num_cnt - 1].x = full_vex_edge[i].x;
			mix_vex_edge[num_cnt - 1].y = full_vex_edge[i].y;
			mix_vex_edge[num_cnt - 1].num = num_cnt;
			mix_vex_edge[num_cnt - 1].num = full_vex_edge[i].num;
			num_cnt++;


		}
		mix_vex_edge[0].vex_size = num_cnt - 1;

		vector<Point2f> after_save;
		vector<int> after_save_num;

		for (int i = 0; i < full_vex_edge[0].edge_size; i++) {

			mix_vex_edge[num_edge_cnt - 1].edge_link_pos[0] = full_vex_edge[i].edge_link_pos[0];
			mix_vex_edge[num_edge_cnt - 1].edge_link_pos[1] = full_vex_edge[i].edge_link_pos[1];

			mix_vex_edge[num_edge_cnt - 1].edge_link_pos[2] = full_vex_edge[i].edge_link_pos[2];
			mix_vex_edge[num_edge_cnt - 1].edge_link_pos[3] = full_vex_edge[i].edge_link_pos[3];

			mix_vex_edge[num_edge_cnt - 1].edge_link_num[0] = full_vex_edge[i].edge_link_num[0];
			mix_vex_edge[num_edge_cnt - 1].edge_link_num[1] = full_vex_edge[i].edge_link_num[1];

			num_edge_cnt++;
		}
		mix_vex_edge[0].edge_size = num_edge_cnt - 1;
	}

	Mat astar_img = pathPlanning->getRGBMap();
	goal_num = pathPlanning->replan_astar(astar_img, mix_vex_edge, path_final_j_inv, cut_path_glob, fixed_path, left_point, is_on_path, robot_next_pos);
	myNameSpace::myImshow("replan astar", astar_img);

	myNameSpace::myWaitKey();
}