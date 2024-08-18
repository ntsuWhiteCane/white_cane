#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "initial_map.hpp"

#ifndef PATH_H
#define PATH_H

class Voronoi_vertex{
	public:
		Voronoi_vertex();
		~Voronoi_vertex();
		// x coordinate of vertex 
		float x;
		// y coordinate of vertex
		float y;
		// number of point( original ID )
		int origin;
		//number of num_match ( new ID -> start from 1)
		int num;
		// the new vertex IDs (a pair -> 2 vertexes) of this edge
		int edge_link_num[2];

		// the original vertex IDs ( a pair -> 2 vertexes) of this edge 
		int edge_link_origin[2];

		// edge_link_pos[0] -> start_point.x of this edge
		// edge_link_pos[1] -> start_point.y of this edge
		// edge_link_pos[2] -> end_point.x of this edge
		// edge_link_pos[3] -> end_point.y of this edge
		float edge_link_pos[4];

		// total vextex amount
		int vex_size;

		// total edge amount
		int edge_size;
	private:
};

class PathPlanning: public InitialMap{
	typedef std::vector<std::vector<double>> Matrix;
	public:
		PathPlanning(cv::Mat &src);
		~PathPlanning();
		void voronoi(cv::Mat &outputImage, Voronoi_vertex *vertex_edge);
		void delaunay(cv::Mat &outputImage);
		// return goal position number
		int replan_astar(cv::Mat &outputImg, Voronoi_vertex *vex_edge, std::vector<int> &path_final_j_inv, std::vector<int> &cut_path_glob, std::vector<int> &fixed_path, cv::Point2f left_point, int is_on_path, cv::Point2f &next_pos);
		/*
		void cubic_spline(int thickness = 1);
		void cubic(std::vector<double> x, std::vector<double> y,
					std::vector<double> *a, std::vector<double> *b,
					std::vector<double> *c, std::vector<double> *d);
		*/
	protected:	
		cv::Rect rect;
		cv::Subdiv2D subdiv;
		std::vector<cv::Point2f> voronoi_points;
		
	private:
		bool isThroughObstacle(cv::Point a, cv::Point b, int thickness = 1);
		Matrix getMinor(const Matrix& src, int row, int col);
		double determinant(const Matrix &mat);
		Matrix adjugate(const Matrix& mat);
		Matrix inverse(const Matrix& mat);
		
};



extern Voronoi_vertex full_vex_edge[2000], roi_vex_edge[2000], temp_vex_edge[800], mix_vex_edge[2000], little_vex_edge[800];

#endif