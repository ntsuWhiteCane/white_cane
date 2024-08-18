#ifndef WHITE_CANE
#define WHITE_CANE

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "initial_map.hpp"
#include "path.hpp"


class WhiteCane{
	public:
		WhiteCane(cv::Mat map);
		~WhiteCane();
		void run();
	private:
		cv::Mat img;
		cv::Mat img_rgb;
		PathPlanning *pathPlanning;
		Voronoi_vertex full_vex_edge[2000], roi_vex_edge[2000], temp_vex_edge[800], mix_vex_edge[2000], little_vex_edge[800];
		std::vector<int> path_final_j_inv;
		std::vector<int> cut_path_glob;
		std::vector<int> fixed_path;
		int is_on_path = 0;
		int fixed_path_cnt = 0;	
		cv::Point2f left_point;
		int goal_num;
		cv::Point2f after_robot_pos, current_robot_pos, robot_next_pos;
		int obs_px = 0, last_obs_px = 9, obs_py = 0, last_obs_py = 9, cnt = 1;

};

#endif