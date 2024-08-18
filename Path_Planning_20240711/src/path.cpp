#include <algorithm>
#include <iostream>
#include <vector>
#include <eigen3\Eigen\Dense>
// #include <fstream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "main.hpp"
#include "path.hpp"

using namespace std;
using namespace cv;
typedef vector<vector<double>> Matrix;

PathPlanning::PathPlanning(Mat &src) : InitialMap(src){
    this->img = src;
    cvtColor(img, img_rgb, COLOR_GRAY2BGR);
}

PathPlanning::~PathPlanning(){

}
// get dalaunay triangle
void PathPlanning::delaunay(Mat &outputImage){
	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);

	Scalar delaunay_color(255, 0, 0);

	//triangleList is the triangle vertex of all delaunay triangles in the image
	// --> ((x1, y1, x2, y2, x3, y3) ...)
	//      ^first triangle         
	vector<Vec6f> triangleList;
	// temporary points for draw delaunay
	vector<Point> tmp_point(3);

	subdiv = Subdiv2D(rect);
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++){
		subdiv.insert(*it);
	}
	subdiv.getTriangleList(triangleList);

	outputImage = img_rgb;
	for (size_t i = 0; i < triangleList.size(); i++){
		Vec6f tmp_triangle = triangleList[i];
		tmp_point[0] = Point(cvRound(tmp_triangle[0]), cvRound(tmp_triangle[1]));
		tmp_point[1] = Point(cvRound(tmp_triangle[2]), cvRound(tmp_triangle[3]));
		tmp_point[2] = Point(cvRound(tmp_triangle[4]), cvRound(tmp_triangle[5]));

		// Draw rectangles completely inside the image.
		if (rect.contains(tmp_point[0]) && rect.contains(tmp_point[1]) && rect.contains(tmp_point[2])){
			// opencv 3.4.1 is CV_LINE, opencv 4.5.5 is LINE_AA
			line(outputImage, tmp_point[0], tmp_point[1], delaunay_color, 1, LINE_AA, 0);
			line(outputImage, tmp_point[1], tmp_point[2], delaunay_color, 1, LINE_AA, 0);
			line(outputImage, tmp_point[2], tmp_point[0], delaunay_color, 1, LINE_AA, 0);
		}
	}

	
}

// draw voronoi
void PathPlanning::voronoi(Mat &outputImage, Voronoi_vertex *vertex_edge){
	// https://docs.opencv.org/3.4/df/dbf/classcv_1_1Subdiv2D.html
	// https://www.twblogs.net/a/5e4fda73bd9eee101df8a0a1	

	// The facets are polygons formed by the vertices of the Voronoi diagram, which are the circumcenters of Delaunay triangles.
	// example: <<[-1200, -1200], [23, 20], [34, 20]>, <[231, 251], [342, 360], [324, 300], [311, 303]>, ...>
	vector<vector<Point2f> > facets;
	// centers are points of delaunay triangle points
	// same as the triangleList but doesn't group
	vector<Point2f> centers;
	// temporary points of "a" polygon
	// ifacet is Point not Point2f --> can draw in image 
	vector<Point> ifacet;
	
	Point2f tmp_point;
	// vector<vector<Point>> ifacets(1);

	int channel_num = 0;
	uint8_t* pixelPtr = (uint8_t*)img.data;
	Scalar_<uint8_t> bgrPixel;

	Size size = img.size();
	Rect rect(0, 0, size.width, size.height);
	subdiv = Subdiv2D(rect);
	cout << "size: " << points.size() << endl;
	for (vector<Point2f>::iterator it = points.begin(); it != points.end(); it++){
		subdiv.insert(*it);
	}

	// get Voronoi infomation to facets and centers
	subdiv.getVoronoiFacetList(vector<int>(), facets, centers);
	
	channel_num = img.channels();
	img_rgb.copyTo(outputImage);
	for (size_t i = 0; i < facets.size(); ++i){

		ifacet.resize(facets[i].size());

		for (size_t j = 0; j < facets[i].size(); ++j) {
			// ifacte is Point2 -> for draw
			// tmp_point is Point2f -> for compute
			ifacet[j] = facets[i][j];
			tmp_point = facets[i][j];

			// check the point doesn't repeat
			for (int k = 0; k < voronoi_points.size(); ++k){
				if (voronoi_points[k] == tmp_point){
					// if repeat, let point out the image size
					tmp_point.x = -1;
					break;
				}
			}
			// check the point is in the image
			if (tmp_point.x < img.size().width && tmp_point.y < img.size().height && tmp_point.x > 0 && tmp_point.y > 0) {
				// same as img.at<uint_8>(ifacet[j].x, ifacet[j].y);
				// but it is faster (usuallay)
				bgrPixel.val[0] = pixelPtr[ ifacet[j].y*img.cols*channel_num + ifacet[j].x*channel_num + 0 ];
				// circle(img_rgb, Point(tmp_point.x, tmp_point.y), 5, Scalar(255, 0, 0), -1);
				// remove the point that is in obstacle
				if (bgrPixel.val[0] == 255){
					voronoi_points.push_back(tmp_point);
					// draw the voronoi points
					// circle(outputImage, Point(tmp_point.x, tmp_point.y), 5, Scalar(255, 0, 0), -1);
				}
			}
		}
		// draw the polygon
		// polylines(outputImage, ifacet, true, Scalar(0, 0, 255), 1, LINE_AA, 0);
		
		// circle(outputImage, centers[i], 3, Scalar(), CV_FILLED, CV_AA, 0);
	}
	cout << endl;
	cout << "facets.size(): " << facets.size() << endl;
	cout << "voronoi_points.size(): " << voronoi_points.size() << endl;
	cout << endl;

	// Generate generalized Voronoi

	// starting point of edge, end point coordinate start_x = v[0], start_y = v[1], end_x = v[2], end_y = v[3].
	// doesn't use.
	vector<Vec4f> edgeList;
	// edge ID of Delaunay triangle
	std::vector<int>  leadingEdgeList;
	subdiv.getEdgeList(edgeList); 
	subdiv.getLeadingEdgeList(leadingEdgeList); 
	
	// delaunay triangle edge ID
	int outer_edges[3];

	// Voronoi edge ID
	int rotate_edges[3];

	// tmp rotate_edges to get voronoi distination vertex 
	int next_rotate_edges[3];

	// record the two point distance.
	vector<double> two_dist;
	vector<double> dist_final;
	//record the two point that is connect
	vector<Point2f> two_match, match_final;

	// vector<int> t1, t2, t3;

	// NOTE: The edge ID of subdiv contains both Delaunay triangle edge IDs and Voronoi Diagram edge IDs.
	// For example, IDs: 12, 212, and 310 could represent a set of Delaunay triangle edge IDs, 
	// while IDs: 13, 213, and 311 could correspond to Voronoi Diagram edge IDs.
	// This is just an example and not indicative of real cases.
	// I do not guarantee that a Voronoi Diagram edge that is perpendicular to a Delaunay edge will have an ID that is incremented by 1 compared to the Delaunay triangle edge.
	
	for (int i = 0; i< leadingEdgeList.size(); ++i){
		// get "an" edge ID of single Delaunay triangle
		outer_edges[0] = leadingEdgeList[i]; 
		// get the other edge ID of triangle (counterclockwise to the outer_edges[0]) (triangle has 3 edge -> 3 ID)
		outer_edges[1] = subdiv.getEdge(outer_edges[0], Subdiv2D::NEXT_AROUND_LEFT);
		// get last edge ID of this triangle
		outer_edges[2] = subdiv.getEdge(outer_edges[1], Subdiv2D::NEXT_AROUND_LEFT);

		// in Subdiv2D class the 0, 1, 2, 3 are not in image
		// 0 is a virtual edge
		// 1, 2, 3 are outside triangle
		// https://www.twblogs.net/a/5e4fda73bd9eee101df8a0a1 
		if (subdiv.edgeOrg(outer_edges[0]) < 4 && subdiv.edgeOrg(outer_edges[1]) < 4 && subdiv.edgeOrg(outer_edges[2]) < 4){
			continue;
		}

		// get the Voroni edge
		
		for (int k = 0; k< 3; ++k) {
			// get the Voroni edge that perpendicular to this Delaunay edge
			rotate_edges[k] = subdiv.rotateEdge(outer_edges[k], 1);
			// nextEdge=getEdge(NEXT_AROUND_ORG) next edge of Voronoi starting point counterclockwise 	
			next_rotate_edges[k] = subdiv.nextEdge(rotate_edges[k]);
		}

		for (int k = 0; k< 3; ++k){
			// get a first vertex point from Voronoi edge.
			Point2f vertex_org = subdiv.getVertex(subdiv.edgeOrg(rotate_edges[k]));
			// get the other vertex point from Voronoi edge.
			Point2f vertex_dst = subdiv.getVertex(subdiv.edgeDst(next_rotate_edges[k]));
			bool flag = false;
			bool org_flag = false;
			bool dst_flag = false;

			for (int j = 0; j< voronoi_points.size(); ++j){
				Point2f point_j = voronoi_points[j];
				//To check if vertex_org is in the voronoi_points array.
				if (cvRound(point_j.x) == cvRound(vertex_org.x) && cvRound(point_j.y) == cvRound(vertex_org.y)) {
					org_flag = true;
				}
				// To check if vertex_dst is in the voronoi_points array.
				if (cvRound(point_j.x) == cvRound(vertex_dst.x) && cvRound(point_j.y) == cvRound(vertex_dst.y)) {
					dst_flag = true;
				}	
			}
			if (!(org_flag && dst_flag)){
				continue;
			}
			// check the the 2 points of voroni edge will not the same point
			// 2 points of voroni edge is the same point never happend in my case
			flag = (subdiv.edgeOrg(rotate_edges[k]) != subdiv.edgeDst(next_rotate_edges[k]));
			// next compute will take long time 
			// if there can break then break
			if (!flag){
				continue;
			}
			flag = flag & (!isThroughObstacle(Point(cvRound(vertex_org.x), cvRound(vertex_org.y)), Point(cvRound(vertex_dst.x), cvRound(vertex_dst.y))));
			if (flag) {
				// // The distance between two points on the Voronoi edge
				// float distance = sqrt(pow(vertex_org.x - vertex_dst.x, 2) + pow(vertex_org.y - vertex_dst.y, 2));
				// two_dist.push_back(distance);

				// Record the starting point of the Voronoi edge and the end number of the next edge
				two_match.push_back(Point2f(subdiv.edgeOrg(rotate_edges[k]), subdiv.edgeDst(next_rotate_edges[k])));
				line(outputImage, vertex_org, vertex_dst, Scalar(255, 0, 255), 1, LINE_4);
			}

		}
		
	}
	
	// now your final_match will like this -> [[120, 30], [390, 43], [12, 30], ...]
	// but we want to let this array like this form [[1, 2], [3, 4], [5, 2], ...]
	// i don't really know why you need to do that. qq

	vector<int> num_match;
	for (int i = 0; i < two_match.size(); i++) {
		int match_p = 0;
		for (int j = 0; j < match_final.size(); j++) {
			// A -> B is same as B -> A
			if (match_final[j].x == two_match[i].y && match_final[j].y == two_match[i].x) {
				match_p = 1;
				break;
			}
		}
		if (match_p == 0) {
			// put in the edge if not the same.
			// dist_final.push_back(two_dist[i]);
			match_final.push_back(two_match[i]);
			num_match.push_back(two_match[i].x);
			num_match.push_back(two_match[i].y);
		}
	}

	// remove the duplicate element
	// num_match -> [vertex ID1, vertex ID2, ...]
	//               0           1         , ...
	sort(num_match.begin(), num_match.end());
	num_match.erase( unique( num_match.begin(), num_match.end() ), num_match.end() );
	
	cout << "two_match.size(): " << two_match.size() << endl;
	cout << "match_final.size(): " << match_final.size() << endl;
	cout << "num_match.size(): " << num_match.size() << endl;


	// new ID of match_final 
	vector<Point2f> num_change;
	
	int num_x, num_y;
	// get the new ID of match_final to num_change
	for (int i = 0; i < match_final.size(); i++) {
		for (int j = 0; j < num_match.size(); j++) {
			// new ID is start from 1
			// i don't now why you need to do that:)
			if (match_final[i].x == num_match[j]) {
				num_x = j+1;
			}
			if (match_final[i].y == num_match[j]) { //Whether the starting point number of the voronoi edge of num_match is the same as the end point number of match_final
				num_y = j+1;
			}

		}
		num_change.push_back(Point2f(num_x, num_y));
	}
	// put the infomation to the voronoi object
	//original plan
	
	cout << "num_match.size(): " << num_match.size() << endl;
	for(int i = 0; i < num_match.size(); i++) {
		vertex_edge[i].origin = num_match[i];
		vertex_edge[i].num = i+1;
		vertex_edge[i].x = subdiv.getVertex(num_match[i]).x; 
		vertex_edge[i].y = subdiv.getVertex(num_match[i]).y;
	}
	cout << "num_change.size(): " << num_change.size() << endl;
	for (int i = 0; i < num_change.size(); i++) {
		vertex_edge[i].edge_link_num[0] = num_change[i].x;  
		vertex_edge[i].edge_link_num[1] = num_change[i].y;  
		vertex_edge[i].edge_link_origin[0] = match_final[i].x; 
		vertex_edge[i].edge_link_origin[1] = match_final[i].y; 
		vertex_edge[i].edge_link_pos[0] = subdiv.getVertex(match_final[i].x).x; 
		vertex_edge[i].edge_link_pos[1] = subdiv.getVertex(match_final[i].x).y;
		vertex_edge[i].edge_link_pos[2] = subdiv.getVertex(match_final[i].y).x;
		vertex_edge[i].edge_link_pos[3] = subdiv.getVertex(match_final[i].y).y;
	}
	vertex_edge[0].vex_size = num_match.size();
	vertex_edge[0].edge_size = num_change.size();	
	
	//re-plan section
	// else {  
	// 	for (int i = 0; i < num_match.size(); i++) {
	// 		roi_vex_edge[i].origin = num_match[i];
	// 		roi_vex_edge[i].num = i+1;
	// 		roi_vex_edge[i].x = subdiv.getVertex(num_match[i]).x;
	// 		roi_vex_edge[i].y = subdiv.getVertex(num_match[i]).y;
	// 	}

	// 	for (int i = 0; i < num_change.size(); i++) {
	// 		roi_vex_edge[i].edge_link_num[0] = num_change[i].x;
	// 		roi_vex_edge[i].edge_link_num[1] = num_change[i].y;
	// 		roi_vex_edge[i].edge_link_origin[0] = match_final[i].x;
	// 		roi_vex_edge[i].edge_link_origin[1] = match_final[i].y;
	// 		roi_vex_edge[i].edge_link_pos[0] = subdiv.getVertex(match_final[i].x).x;
	// 		roi_vex_edge[i].edge_link_pos[1] = subdiv.getVertex(match_final[i].x).y;
	// 		roi_vex_edge[i].edge_link_pos[2] = subdiv.getVertex(match_final[i].y).x;
	// 		roi_vex_edge[i].edge_link_pos[3] = subdiv.getVertex(match_final[i].y).y;
	// 	}
	// 	roi_vex_edge[0].vex_size = num_match.size();
	// 	roi_vex_edge[0].edge_size = num_change.size();
	// }

}


bool PathPlanning::isThroughObstacle(Point a, Point b, int thickness){
	int min_x = MIN(a.x, b.x);
	int min_y = MIN(a.y, b.y);
	int max_x = MAX(a.x, b.x);
	int max_y = MAX(a.y, b.y);

	// if(min_x < 0 || min_y < 0 || max_x >= img.cols || max_y > img.rows){
	// 	return true;
	// }

	Mat tmp_img(img.size(), CV_8UC1, Scalar(255));
	line(tmp_img, a, b, Scalar(0), thickness);

	int channel_num1 = img.channels();
	int channel_num2 = tmp_img.channels();
	uint8_t *pixelPtr1 = (uint8_t*)img.data;
	uint8_t *pixelPtr2 = (uint8_t*)tmp_img.data;
	Scalar_<uint8_t> bgrPixel1, bgrPixel2;
		
	for (int i = min_x; i<= max_x; ++i){
		for (int j = min_y; j<= max_y; ++j){
			bgrPixel1.val[0] = pixelPtr1[ j*img.cols*channel_num1 + i*channel_num1 + 0 ];
			bgrPixel2.val[0] = pixelPtr2[ j*tmp_img.cols*channel_num2 + i*channel_num2 + 0 ];
			if (!(bgrPixel1.val[0] | bgrPixel2.val[0])){
				return true;
			}
		}
	}
	return false;
}

int PathPlanning::replan_astar(Mat &outputImg, Voronoi_vertex *vex_edge, vector<int> &path_final_j_inv, vector<int> &cut_path_glob, vector<int> &fixed_path, Point2f left_point, int is_on_path, Point2f &next_pos){
	int goal_num;
	// close_list is an array used to store whether a node has been traversed during the execution of the A* algorithm. 
	int close_list[2000] = {0};

	// heur is an array used to store the distances from each vertex to the goal.
	double heur[2000] = {0};

	// dist_path is used to store the vertex from which the current vertex was reached.
	// e.g dist_path[300] = 129 indicates that in the current A* algorithm execution, the vertex with ID 300 was reached from the vertex with ID 129. 
	// In other words, the predecessor of the vertex with ID 300 is the vertex with ID 129.
	int dis_path[2000] = {0};

	int init, goal, cost, after_num;
	float x, y, g_num, f_num, x2, y2, g2_num, f2_num;
	vector<float> cell, cell_h, action, invpath, path;
	vector<int> delta;
	vector<int> open_list, priority, near_point;
	vector<float> dist_goal, dist_init, tmp_dist;
	int open_p = 0;
	int small_pos = 0;
	bool found = false, resign = false;
	Point2f init_pos, goal_pos;
	std::vector<float>::iterator smallest;
	Point2f init_goal_toward;
	vector<int> init_num_vec, init_num_vec2;
	Mat ori_astar_show;

	img_rgb.copyTo(ori_astar_show);
	img_rgb.copyTo(outputImg);
	// start_point
	init_pos = robot_position;
	// end_point
	goal_pos = goal_position;	

	double start_time, end_time;

	start_time = clock();
	// get toward vector
	// doesn't use. and this is seem to wrong ->
	// init_goal_toward = Point2f(goal_pos.x - init_pos.x, goal_pos.y - init_pos.y); 
	init_goal_toward = Point2f(init_pos.x - goal_pos.x, init_pos.y - goal_pos.y);

	float each_init_dis = 0;
	int is_break = 0;

	// Problem: Avoid taking routes that cross moving obstacles  (re-planning)

	// three_min_dist_p are 3 points that closest to the starting/end point. 
	vector<int> three_min_dis_p, three_min_dis_p2;
	int i_init, j_init, i_back, j_back, is_line_obs = 0;
	
	//replan
	if (is_on_path == 1) {   
		for (int i = 0; i < vex_edge[0].vex_size; i++) {
			if (vex_edge[i].x > left_point.x && vex_edge[i].y > left_point.y && vex_edge[i].x < left_point.x + 80 && vex_edge[i].y < left_point.y + 80) {
				dist_init.push_back(sqrt(pow(vex_edge[i].x - init_pos.x, 2) + pow(vex_edge[i].y - init_pos.y, 2)));
				init_num_vec.push_back(i);
			}

			each_init_dis = sqrt(pow(vex_edge[i].x - init_pos.x, 2) + pow(vex_edge[i].y - init_pos.y, 2));

			if (each_init_dis < 2){
				init = vex_edge[i].num;
				is_break = 1;
				break;
			}
		}
		//avoid identical points
		int same_vex = 1, i_add = 1;
		for (int i = 0; i < init_num_vec.size(); i++) {
			for (int j = 0 + i_add; j < init_num_vec.size(); j++) {
				same_vex = 0;
				if (abs(vex_edge[init_num_vec[i]].x - vex_edge[init_num_vec[j]].x) < 0.1 && abs(vex_edge[init_num_vec[i]].y - vex_edge[init_num_vec[j]].y) < 0.1) {
					same_vex = 1;
					break;
				}
			}
			if (same_vex == 0) {
				tmp_dist.push_back(sqrt(pow(vex_edge[init_num_vec[i]].x - init_pos.x, 2) + pow(vex_edge[init_num_vec[i]].y - init_pos.y, 2)));
				init_num_vec2.push_back(init_num_vec[i]);
			}
			i_add++;
		}
		dist_init.clear();
		init_num_vec.clear();
		dist_init = tmp_dist;
		init_num_vec = init_num_vec2;
	}
	else {
		//Problem: Avoid taking routes that cross moving obstacles  (re-planning)
		for (int i = 0; i < vex_edge[0].vex_size; i++) {
			// compute the every vertex distance of initial_position
			each_init_dis = sqrt(pow(vex_edge[i].x - init_pos.x, 2) + pow(vex_edge[i].y - init_pos.y, 2));
			dist_init.push_back(each_init_dis);
			// get this time id
			init_num_vec.push_back(i);
			// if the vertex is too close the initial point then break
			// find the Voronoi point closest to the end point

			// The starting point and the nearest Voronoi point are less than 2
			if (each_init_dis < 2) {
				init = vex_edge[i].num;
				is_break = 1;
				break;
			}
		}
	}
	cout << "is_break:" << is_break << endl;

	/////////create a node connected to the starting point///////////
	if (is_break == 0) {
		for (int i = 0; i< 3; ++i){
			// smallest is the mininmum distance of some vertex to initial point
			smallest = std::min_element(std::begin(dist_init), std::end(dist_init));
			// find the minimum distance position in dist_init array 
			small_pos = std::distance(std::begin(dist_init), smallest);
			// our id is start from 1
			// putting the 3 points that closest point of the starting points
			three_min_dis_p.push_back(init_num_vec[small_pos] + 1);

			// let closest point of starting points be a large number 
			// -> to find next closest point
			dist_init[small_pos] = 99999;
		}

		dist_init.clear();
		for (int i = 0; i < three_min_dis_p.size(); i++) {
			// put the a* algo cost of the 3 points that closest point of the starting points 
			// to find the most avaliable point that starting point connect
			dist_init.push_back(sqrt(pow(vex_edge[three_min_dis_p[i] - 1].x - init_pos.x, 2) + pow(vex_edge[three_min_dis_p[i] - 1].y - init_pos.y, 2)) + sqrt(pow(vex_edge[three_min_dis_p[i] - 1].x - goal_pos.x, 2)
				+ pow(vex_edge[three_min_dis_p[i] - 1].y - goal_pos.y, 2)));
		}

		for (int n = 0; n< 3; ++n){
			smallest = std::min_element(std::begin(dist_init), std::end(dist_init));
			small_pos = std::distance(std::begin(dist_init), smallest);
			// get roi 
			i_init = MIN(vex_edge[three_min_dis_p[small_pos] - 1].x, init_pos.x);
			j_init = MIN(vex_edge[three_min_dis_p[small_pos] - 1].y, init_pos.y);
			i_back = MAX(vex_edge[three_min_dis_p[small_pos] - 1].x, init_pos.x);
			j_back = MAX(vex_edge[three_min_dis_p[small_pos] - 1].y, init_pos.y);
			is_line_obs = 0;
			for (int i = i_init; i <= i_back; i++) {
				for (int j = j_init; j <= j_back; j++) {	
					if (img.at<uint8_t>(j, i) == 0) {
						is_line_obs = 1;
						break;
					}
				}
				if (is_line_obs == 1) {
					break;
				}
			}
			// if the closest point has obstacle in roi then chose the next closet point of starting point
			// if first and second time both have obstcle in roi, no matter is third time has obstcle in roi 
			// chose the third point
			if (is_line_obs == 1 && n< 2) {
				dist_init[small_pos] = 9999;
				smallest = std::min_element(std::begin(dist_init), std::end(dist_init));
				small_pos = std::distance(std::begin(dist_init), smallest);
			}
			else{
				break;
			}
		}

		// add initial_pos into vex_edge
		vex_edge[vex_edge[0].vex_size].num = vex_edge[0].vex_size + 1;
		vex_edge[vex_edge[0].vex_size].x = init_pos.x;
		vex_edge[vex_edge[0].vex_size].y = init_pos.y;
		vex_edge[0].vex_size++;

		// link the starting point to link point
		vex_edge[vex_edge[0].edge_size].edge_link_num[0] = vex_edge[vex_edge[0].vex_size - 1].num;
		vex_edge[vex_edge[0].edge_size].edge_link_num[1] = three_min_dis_p[small_pos];
		vex_edge[0].edge_size++;

		init = vex_edge[vex_edge[0].vex_size - 1].num;
	}

	/////////Create a node connected to the starting point///////////

	//////Determine whether the end point is on the node//////
	vector<int> goal_num_vec;
	float each_goal_dis = 0;
	is_break = 0;
	is_line_obs = 0;
	for (int i = 0; i < vex_edge[0].vex_size; i++) {
		each_goal_dis = sqrt(pow(vex_edge[i].x - goal_pos.x, 2) + pow(vex_edge[i].y - goal_pos.y, 2));
		goal_num_vec.push_back(i);
		dist_goal.push_back(sqrt(pow(vex_edge[i].x - goal_pos.x, 2) + pow(vex_edge[i].y - goal_pos.y, 2)));

		if (each_goal_dis < 1) {
			goal = vex_edge[i].num;
			goal_num = vex_edge[i].num;
			is_break = 1;
			break;
		}
	}
	//////Determine whether the end point is on the node//////

	//////Create a node connected to the end point//////
	if (is_break == 0) {
		for (int i = 0; i< 3; ++i){
			smallest = std::min_element(std::begin(dist_goal), std::end(dist_goal));
			small_pos = std::distance(std::begin(dist_goal), smallest);
			three_min_dis_p2.push_back(goal_num_vec[small_pos] + 1);
			dist_goal[small_pos] = 99999;
		}
		dist_goal.clear();

		for (int i = 0; i < three_min_dis_p2.size(); i++) {
			dist_goal.push_back(sqrt(pow(vex_edge[three_min_dis_p2[i] - 1].x - init_pos.x, 2) + pow(vex_edge[three_min_dis_p2[i] - 1].y - init_pos.y, 2)) + sqrt(pow(vex_edge[three_min_dis_p2[i] - 1].x - goal_pos.x, 2)
				+ pow(vex_edge[three_min_dis_p2[i] - 1].y - goal_pos.y, 2)));
		}

		for(int n = 0; n< 3; ++n){
			smallest = std::min_element(std::begin(dist_goal), std::end(dist_goal));
			small_pos = std::distance(std::begin(dist_goal), smallest);

			i_init = MIN(vex_edge[three_min_dis_p2[small_pos] - 1].x, goal_pos.x);
			j_init = MIN(vex_edge[three_min_dis_p2[small_pos] - 1].y, goal_pos.y);
			i_back = MAX(vex_edge[three_min_dis_p2[small_pos] - 1].x, goal_pos.x);
			j_back = MAX(vex_edge[three_min_dis_p2[small_pos] - 1].y, goal_pos.y);	

			is_line_obs = 0;	
			for (int i = i_init; i <= i_back; i++) {
				for (int j = j_init; j <= j_back; j++) {
					if (img.at<uint8_t>(j, i) == 0) {
						is_line_obs = 1;
						break;
					}
				}
				if (is_line_obs == 1) {
					break;
				}
			}
			if (is_line_obs == 1 && n< 2) {
				dist_goal[small_pos] = 9999;
				smallest = std::min_element(std::begin(dist_goal), std::end(dist_goal));
				small_pos = std::distance(std::begin(dist_goal), smallest);
			}
			else{
				break;
			}
		}

		vex_edge[vex_edge[0].vex_size].num = vex_edge[0].vex_size + 1;
		vex_edge[vex_edge[0].vex_size].x = goal_pos.x;
		vex_edge[vex_edge[0].vex_size].y = goal_pos.y;
		vex_edge[0].vex_size++;

		vex_edge[vex_edge[0].edge_size].edge_link_num[0] = vex_edge[vex_edge[0].vex_size - 1].num;
		vex_edge[vex_edge[0].edge_size].edge_link_num[1] = three_min_dis_p2[small_pos];
		vex_edge[0].edge_size++;

		goal = vex_edge[vex_edge[0].vex_size - 1].num;
		goal_num = vex_edge[vex_edge[0].vex_size - 1].num;

	}
	//////Create a node connected to the end point//////

	cout << "init:" << init << endl;
	cout << "goal:" << goal << endl;


	for (int i = 0; i < vex_edge[0].vex_size; i++) {
		// compute every distance from vetex to end.
		heur[i + 1] = sqrt(pow(vex_edge[i].x - vex_edge[goal - 1].x, 2) + pow(vex_edge[i].y - vex_edge[goal - 1].y, 2));
	}


	///////compute A*///////////
	close_list[init] = 1;
	int cell_num = 0;
	g_num = 0;
	f_num = g_num + heur[init];
	cell_num = init;
	cell.push_back(f_num);
	cell.push_back(g_num);
	cell.push_back(cell_num);
	int cnt = 0, now_num = 0;
	int cell_small;
	//std::vector<float>::iterator smallest;

	while (found != true && resign != true){

		if (abs(goal_pos.x - init_pos.x) < 1 && abs(goal_pos.y - init_pos.y) < 1) {
			break;
		}

		if (cell.empty()) {
			resign = true;
			cout << "fail" << endl;
		}
		else {
			for (int i = 0; i < cell.size() / 3; i++) {
				cell_h.push_back(cell[3 * i]);
			}
			// find the smallest cost in graph
			smallest = std::min_element(std::begin(cell_h), std::end(cell_h));
			// get the smallest cost edge position in cell_h array
			cell_small = std::distance(std::begin(cell_h), smallest);
			// vertex ID	
			cell_num = cell[3 * cell_small + 2];
			// cost from start to vertex 
			g_num = cell[3 * cell_small + 1];
			// cost of vertex
			f_num = cell[3 * cell_small];
			
			x = vex_edge[cell_num - 1].x;
			y = vex_edge[cell_num - 1].y;

			// After obtaining the next algorithm running vertex, remove the vertex from the vertices to be calculated.
			cell.erase(cell.begin() + 3 * cell_small, cell.begin() + 3 * cell_small + 3);

			cnt++;
			cell_h.clear();
			delta.clear();
			now_num = 0;

			if (cell_num == goal) {
				found = true;
			}
			else {
				for (int i = 0; i < vex_edge[0].edge_size; i++) {
					// Get the vertices connected to the current operation
					if (cell_num == vex_edge[i].edge_link_num[0]) {
						if (vex_edge[i].edge_link_num[0] != vex_edge[i].edge_link_num[1]) {
							delta.push_back(vex_edge[i].edge_link_num[1]);
						}
					}
					else if (cell_num == vex_edge[i].edge_link_num[1]) {
						if (vex_edge[i].edge_link_num[0] != vex_edge[i].edge_link_num[1]) {
							delta.push_back(vex_edge[i].edge_link_num[0]);
						}

					}
				}

				for (int i = 0; i < delta.size(); i++) {
					//cout << "delta:" << delta[i] << endl;
					x2 = vex_edge[delta[i] - 1].x;
					y2 = vex_edge[delta[i] - 1].y;
					cost = sqrt(pow((x2 - x), 2) + pow((y2 - y), 2));
					
					if (close_list[delta[i]] == 0) {
						g2_num = g_num + cost;
						//g2_num = cost;
						f2_num = g2_num + heur[delta[i]];	
						close_list[delta[i]] = 1;
						// dis_path[int(num_match[j].x)] = change_number(cell_num, num_match, 2);
						// this code means the vertex that ID = delta[i] is link from cell_num(and this link is minimum cost at that moment)
						dis_path[delta[i]] = cell_num;
						cell.push_back(f2_num);
						cell.push_back(g2_num);
						cell.push_back(delta[i]);
					}

				}


			}
		}
	}
	///////compute A*///////////

	///////find path////////
	vector<int> path_final_j;
	path_final_j.clear();
	path_final_j_inv.clear();
	
	path_final_j.push_back(goal);

	int next_path = 0;
	next_path = dis_path[goal];
	while (next_path != 0){
		path_final_j.push_back(next_path);
		next_path = dis_path[next_path];
	}

	for (int i = path_final_j.size() - 1; i >= 0; i--) {
		path_final_j_inv.push_back(path_final_j[i]);
	}
	///////find path////////
	

	///////draw path planning//////////
	if (resign == true) {
		//line(img, Point2f(vex_edge[init - 1].x, vex_edge[init - 1].y), Point2f(vex_edge[goal - 1].x, vex_edge[goal - 1].y), Scalar(255, 100, 255), 1, CV_AA);
		line(outputImg, Point2f(vex_edge[init - 1].x, vex_edge[init - 1].y), Point2f(goal_pos.x, goal_pos.y), Scalar(255, 100, 255), 1, LINE_AA);
	}
	else {
		if (abs(goal_pos.x - init_pos.x) < 1 && abs(goal_pos.y - init_pos.y) < 1) {

		}
		else {
			for (int i = 0; i < path_final_j.size() - 1; i++) {

				//line(img, Point2f(vex_edge[path_final_j[i] - 1].x, vex_edge[path_final_j[i] - 1].y), Point2f(vex_edge[path_final_j[i + 1] - 1].x, vex_edge[path_final_j[i + 1] - 1].y), Scalar(255, 0, 0), 1, CV_AA);
				line(ori_astar_show, Point2f(vex_edge[path_final_j[i] - 1].x, vex_edge[path_final_j[i] - 1].y), Point2f(vex_edge[path_final_j[i + 1] - 1].x, vex_edge[path_final_j[i + 1] - 1].y), Scalar(255, 0, 0), 1, LINE_AA);

			}
			//line(img, Point2f(vex_edge[path_final_j[0] - 1].x, vex_edge[path_final_j[0] - 1].y), Point2f(goal_pos.x, goal_pos.y), Scalar(255, 0, 0), 1, CV_AA);
			line(ori_astar_show, Point2f(vex_edge[path_final_j[0] - 1].x, vex_edge[path_final_j[0] - 1].y), Point2f(goal_pos.x, goal_pos.y), Scalar(255, 0, 0), 1, LINE_AA);

			for (int i = 0; i < path_final_j.size(); i++) {
				circle(ori_astar_show, Point2f(vex_edge[path_final_j[i] - 1].x, vex_edge[path_final_j[i] - 1].y), 3, Scalar(0, 0, 250), FILLED);
				//circle(img, Point2f(vex_edge[cut_path[i] - 1].x, vex_edge[cut_path[i] - 1].y), 3, Scalar(100, 0, 255), FILLED);
			}
		}
	}
	circle(ori_astar_show, Point2f(vex_edge[init - 1].x, vex_edge[init - 1].y), 4, Scalar(0, 0, 255), FILLED);
	circle(ori_astar_show, Point2f(vex_edge[path_final_j[0] - 1].x, vex_edge[path_final_j[0] - 1].y), 4, Scalar(0, 255, 0), FILLED);
	myNameSpace::myImshow("ori_astar_show", ori_astar_show);
	///////darw the path planning//////////


	// Reduce walking nodes (but will not speed up calculation time) - count from the starting point
	cut_path_glob.clear();
	float step_path = 0;
	int sub_p = 0;
	vector<int> cut_path, tmp_cut;
	tmp_cut = path_final_j_inv;

	// delete the nodes that too close
	for (int i = 0; i < tmp_cut.size() - 1; i++) {
		step_path = sqrt(pow(vex_edge[tmp_cut[i - sub_p] - 1].x - vex_edge[tmp_cut[i + 1] - 1].x, 2) + pow(vex_edge[tmp_cut[i - sub_p] - 1].y - vex_edge[tmp_cut[i + 1] - 1].y, 2));
		if (step_path > 20 || i == 0) {
			cut_path.push_back(tmp_cut[i]);	
			sub_p = 0;
		}
		else {
			sub_p++;
		}
	}
	// put the goal position ID
	cut_path.push_back(tmp_cut[tmp_cut.size() - 1]);

	tmp_cut.clear();
	tmp_cut = cut_path;
	cut_path.clear();

	Point2f prev_vector(0, 0);
	Point2f current_vector(0, 0);
	float tmp_dot_product = 0;
	float tmp_norm = 1;
	sub_p = 0;
	// delete the node that has same towards
	for (int i = 0; i< tmp_cut.size() - 1; ++i){
		float tmp_x = vex_edge[tmp_cut[i + 1] - 1].x - vex_edge[tmp_cut[i - sub_p] - 1].x;
		float tmp_y = vex_edge[tmp_cut[i + 1] - 1].y - vex_edge[tmp_cut[i - sub_p] - 1].y;
		tmp_norm = sqrt(tmp_x*tmp_x + tmp_y*tmp_y);
		current_vector = Point2f(tmp_x, tmp_y) / tmp_norm;
		tmp_dot_product = current_vector.x * prev_vector.x + current_vector.y * prev_vector.y;
		
		if (tmp_dot_product < 0.9998){
			cut_path.push_back(tmp_cut[i]);
			prev_vector = current_vector;
			sub_p = 0;
		}
		else{
			++sub_p;
		}
	}
	// put the goal position ID
	cut_path.push_back(tmp_cut[tmp_cut.size() - 1]);
	tmp_cut.clear();
	tmp_cut = cut_path;
	cut_path.clear();
	cut_path.push_back(tmp_cut[0]);
	cut_path.push_back(tmp_cut[tmp_cut.size() - 1]);
	// Get shortcuts to start and end points
	for (int i = 0; i< tmp_cut.size(); ++i){
		bool flag = isThroughObstacle(Point(vex_edge[tmp_cut[0] - 1].x, vex_edge[tmp_cut[0] - 1].y), Point(vex_edge[tmp_cut[i] - 1].x, vex_edge[tmp_cut[i] - 1].y), 1/RESOLUTION);
		if(flag && i<= 1){
			cut_path = tmp_cut;
			break;
		}
		if (flag){
			cut_path.clear();
			cut_path.push_back(tmp_cut[0]);
			cut_path.insert(cut_path.end(), tmp_cut.begin()+i-1, tmp_cut.end());
			break;
		}
	}

	tmp_cut.clear();
	tmp_cut = cut_path;
	cut_path.clear();
	cut_path.push_back(tmp_cut[0]);
	cut_path.push_back(tmp_cut[tmp_cut.size() - 1]);

	for(int i = tmp_cut.size()-1; i >= 0; --i){
		bool flag = isThroughObstacle(Point(vex_edge[tmp_cut[tmp_cut.size()-1] - 1].x, vex_edge[tmp_cut[tmp_cut.size()-1] - 1].y), Point(vex_edge[tmp_cut[i] - 1].x, vex_edge[tmp_cut[i] - 1].y), 1/RESOLUTION);
		if(flag && i >= tmp_cut.size()-2){
			cut_path = tmp_cut;
			break;
		}
		if(flag){
			cut_path.clear();
			cut_path.push_back(tmp_cut[tmp_cut.size()-1]);
			cut_path.insert(cut_path.begin(), tmp_cut.begin(), tmp_cut.begin()+i+2);
			break;
		}
	}

	// cut_path.clear();
	cut_path_glob = cut_path;

	for (int i = 0; i < cut_path.size() - 1; i++) {
		line(outputImg, Point2f(vex_edge[cut_path[i] - 1].x, vex_edge[cut_path[i] - 1].y), Point2f(vex_edge[cut_path[i + 1] - 1].x, vex_edge[cut_path[i + 1] - 1].y), Scalar(255, 0, 0), 2, LINE_AA);
	}
	for (int i = 0; i < cut_path.size(); i++) {
		circle(outputImg, Point2f(vex_edge[cut_path[i] - 1].x, vex_edge[cut_path[i] - 1].y), 2, Scalar(200, 0, 255), FILLED);
		cout << Point2f(vex_edge[cut_path[i] - 1].x, vex_edge[cut_path[i] - 1].y) << endl;
	}
	circle(outputImg, Point2f(vex_edge[init - 1].x, vex_edge[init - 1].y), 3, Scalar(0, 0, 255), FILLED);
	circle(outputImg, Point2f(vex_edge[path_final_j[0] - 1].x, vex_edge[path_final_j[0] - 1].y), 3, Scalar(0, 255, 0), FILLED);

	for (int i = 0; i < cut_path.size(); i++) {
		fixed_path.push_back(cut_path[i]);
	}
	//Reduce walking nodes (but will not speed up calculation time) - count from the starting point



	////determind the next node/////
	//if (init != goal) {
	if (path_final_j.size() > 1) {
		if (abs(vex_edge[cut_path[0] - 1].x - init_pos.x) <= 1 && abs(vex_edge[cut_path[0] - 1].y - init_pos.y) <= 1) {
			robot_position.x = vex_edge[cut_path[0] - 1].x;
			robot_position.y = vex_edge[cut_path[0] - 1].y;
			next_pos.x = vex_edge[cut_path[1] - 1].x;
			next_pos.y = vex_edge[cut_path[1] - 1].y;
			// used for the sake of having the same points
			if (abs(robot_position.x - next_pos.x) < 2 && abs(robot_position.y - next_pos.y) < 2) {
				robot_position.x = vex_edge[cut_path[1] - 1].x;
				robot_position.y = vex_edge[cut_path[1] - 1].y;
				next_pos.x = vex_edge[cut_path[2] - 1].x;
				next_pos.y = vex_edge[cut_path[2] - 1].y;
			}
			cout << "now_pos1:" << robot_position << " " << "r_next_pos1:" << next_pos << endl;

		}
		else {
			robot_position.x = init_pos.x;
			robot_position.y = init_pos.y;
			next_pos.x = vex_edge[cut_path[0] - 1].x;
			next_pos.y = vex_edge[cut_path[0] - 1].y;
			cout << "r_now_pos2:" << robot_position << " " << "r_next_pos2:" << next_pos << endl;
		}
	}
	else {
		robot_position.x = vex_edge[cut_path[0] - 1].x;
		robot_position.y = vex_edge[cut_path[0] - 1].y;
		next_pos.x = vex_edge[cut_path[0] - 1].x;
		next_pos.y = vex_edge[cut_path[0] - 1].y;
	}

	////decide the next node/////

	//circle(img, Point2f(aft_robot_pos.x, aft_robot_pos.y), 2, Scalar(30, 0, 255), FILLED);

	myNameSpace::myImshow("replan_astar", outputImg);
	end_time = clock();

	cout << endl << "runing time of replan_a*: " << (end_time - start_time) / CLOCKS_PER_SEC << " S" << endl;
	return goal_num;

}

Matrix PathPlanning::getMinor(const Matrix& src, int row, int col) {
    int n = src.size();
    Matrix minor(n - 1, vector<double>(n - 1));
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1; j++) {
            minor[i][j] = src[i < row ? i : i + 1][j < col ? j : j + 1];
        }
    }
    return minor;
}

double PathPlanning::determinant(const Matrix& mat) {
    int n = mat.size();
    if (n == 1) return mat[0][0];
    if (n == 2) return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    double det = 0;
    for (int j = 0; j < n; j++) {
        det += mat[0][j] * determinant(getMinor(mat, 0, j)) * (j % 2 == 0 ? 1 : -1);
    }
    return det;
}

Matrix PathPlanning::adjugate(const Matrix& mat) {
    int n = mat.size();
    Matrix adj(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            adj[j][i] = determinant(getMinor(mat, i, j)) * ((i + j) % 2 == 0 ? 1 : -1);
        }
    }
    return adj;
}

Matrix PathPlanning::inverse(const Matrix& mat){
	int n = mat.size();
	double det = determinant(mat);
	if (det == 0) throw runtime_error("Matrix is singular and cannot be inverted.");
	Matrix adj = adjugate(mat);
	Matrix inv(n, vector<double>(n));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			// cout << i << ", " << "j" << endl;
			inv[i][j] = adj[i][j] / det;
		}
	}
	return inv;
}

/*
void PathPlanning::cubic(vector<double> x, vector<double> y, vector<double> *a, vector<double> *b, vector<double> *c, vector<double> *d){
	// Matrix inv;
	
	int n = x.size() - 1;
	vector<double> h(n, 0);
	vector<vector<double>> H(n+1, vector<double>(n+1, 0));
	vector<double> Y(n+1, 0);
	vector<double> M(n+1, 0);
	vector<double> A(n, 0);
	vector<double> B(n, 0);
	vector<double> C(n, 0);
	vector<double> D(n, 0);
	
	for(int i = 0; i< n; ++i){
		h[i] = x[i+1] - x[i];
	}
	H[0][0] = 1;
	H[n][n] = 1;
	for(int i = 1; i< n; ++i){
		H[i][i-1] = h[i-1];
		H[i][i] = 2*(h[i-1] + h[i]);
		H[i][i+1] = h[i];
		Y[i] = 6*( (y[i+1] - y[i]) / h[i] - (y[i] - y[i-1]) / h[i - 1]);
	}
	Eigen::MatrixXf inv(n+1, n+1);
	for (int i = 0; i< n+1; ++i){
		for(int j = 0; j< n+1; ++j){
			inv(i, j) = H[i][j];
		}
	}
	inv = inv.inverse();
	
	
	for(int i = 0; i< n+1; ++i){
		double sum = 0;
		for(int j = 0; j< Y.size(); ++j){
			sum += inv(i, j) * Y[j];
			// cout << "inv(" << i << ", " << j << "): " << inv(i, j) << "; " << "Y[" << j << "]: " << Y[j] << "; ";
			// cout << "sum: " << sum << endl;
		}
		M[i] = sum;
		// cout << "M[" << i << "]: " << M[i] << " ";
		// cout << endl;
	}
	
	for(int i = 0; i< n; ++i){
		A[i] = y[i];
		B[i] = ((y[i+1] - y[i])/h[i]) - (M[i]/2 * h[i]) - ((M[i+1] - M[i])/6 * h[i]);
		C[i] = M[i] / 2;
		D[i] = (M[i+1] - M[i]) / (6*h[i]);
	}

	*a = A;
	*b = B;
	*c = C;
	*d = D;


}

void PathPlanning::cubic_spline(int thickness){
	
	int path_arc_length = 0;
	int max_change = 0;
	int change_count = 0;
	Mat tmp_img1;

	x.clear();
	y.clear();
	t.clear();

	for(int i = 0; i< fixed_path.size(); ++i){
		// t.push_back(i);
		x.push_back(mix_vex_edge[fixed_path[i] - 1].x);
		y.push_back(mix_vex_edge[fixed_path[i] - 1].y);
	}

	t.push_back(0);
	for(int i = 0; i< fixed_path.size()-1; ++i){
		t.push_back(t[i] + sqrt( (x[i+1] - x[i])* (x[i+1] - x[i]) + (y[i+1] - y[i])* (y[i+1] - y[i]) ) / (1/RESOLUTION));
	}

	path_arc_length = t[fixed_path.size() - 1] * (1/RESOLUTION);

	max_change = fixed_path.size() * 2;

	cubic(t, x, &x_a_coefficient, &x_b_coefficient, &x_c_coefficient, &x_d_coefficient);
	cubic(t, y, &y_a_coefficient, &y_b_coefficient, &y_c_coefficient, &y_d_coefficient);
	vector<double> tmp_x_a_coefficient, tmp_x_b_coefficient, tmp_x_c_coefficient, tmp_x_d_coefficient;
	vector<double> tmp_y_a_coefficient, tmp_y_b_coefficient, tmp_y_c_coefficient, tmp_y_d_coefficient;

	tmp_x_a_coefficient = x_a_coefficient;
	tmp_x_b_coefficient = x_b_coefficient;
	tmp_x_c_coefficient = x_c_coefficient;
	tmp_x_d_coefficient = x_d_coefficient;

	tmp_y_a_coefficient = y_a_coefficient;
	tmp_y_b_coefficient = y_b_coefficient;
	tmp_y_c_coefficient = y_c_coefficient;
	tmp_y_d_coefficient = y_d_coefficient;

	bool path_danger = true;
	while(path_danger){
		bool tmp_flag = false;

		if(change_count > max_change){
			x_a_coefficient = tmp_x_a_coefficient;
			x_b_coefficient = tmp_x_b_coefficient;
			x_c_coefficient = tmp_x_c_coefficient;
			x_d_coefficient = tmp_x_d_coefficient;

			y_a_coefficient = tmp_y_a_coefficient;
			y_b_coefficient = tmp_y_b_coefficient;
			y_c_coefficient = tmp_y_c_coefficient;
			y_d_coefficient = tmp_y_d_coefficient;
			cout << "avoid fail." << endl;
			break;
		}

		for(int i = 0; i< x_a_coefficient.size(); ++i){
			for(int n = 0; n< 100-1; ++n){
				Point2f p1, p2;
				double dt1 = (t[i] + (t[i+1] - t[i]) / 100 * n) - t[i];
				double dt2 = (t[i] + (t[i+1] - t[i]) / 100 * (n+1)) - t[i];
				p1.x = x_a_coefficient[i] + x_b_coefficient[i]*dt1 + x_c_coefficient[i]*dt1*dt1 + x_d_coefficient[i]*dt1*dt1*dt1;
				p1.y = y_a_coefficient[i] + y_b_coefficient[i]*dt1 + y_c_coefficient[i]*dt1*dt1 + y_d_coefficient[i]*dt1*dt1*dt1;
				p2.x = x_a_coefficient[i] + x_b_coefficient[i]*dt2 + x_c_coefficient[i]*dt2*dt2 + x_d_coefficient[i]*dt2*dt2*dt2;
				p2.y = y_a_coefficient[i] + y_b_coefficient[i]*dt2 + y_c_coefficient[i]*dt2*dt2 + y_d_coefficient[i]*dt2*dt2*dt2;
				// circle(tmp_img1, Point(p1.x, p1.y), 1, Scalar(255, 0, 0), -1);
				if(isThroughObstacle(Point(p1.x, p1.y), Point(p2.x, p2.y), thickness)){
					double sub;
					sub = (t[i+1]-t[i]) * 0.1;
					for(int j = i; j< fixed_path.size()-1; ++j){
						
						t[j+1] -= sub; 
					}

					cubic(t, x, &x_a_coefficient, &x_b_coefficient, &x_c_coefficient, &x_d_coefficient);
					cubic(t, y, &y_a_coefficient, &y_b_coefficient, &y_c_coefficient, &y_d_coefficient);
					change_count++;
					tmp_flag = true;
					path_danger = true;
					break;
				}
				// line(tmp_img1, Point(p1.x, p1.y), Point(p2.x, p2.y), Scalar(0), thickness);
			}
			if(tmp_flag){
				break;
			}
			path_danger = false;
		}
	}

	tmp_img1 = getRGBMap();
	for(int i = 0; i< x_a_coefficient.size(); ++i){
		for(int n = 0; n< 100-1; ++n){
			Point2f p1, p2;
			double dt1 = (t[i] + (t[i+1] - t[i]) / 100 * n) - t[i];
			double dt2 = (t[i] + (t[i+1] - t[i]) / 100 * (n+1)) - t[i];
			p1.x = x_a_coefficient[i] + x_b_coefficient[i]*dt1 + x_c_coefficient[i]*dt1*dt1 + x_d_coefficient[i]*dt1*dt1*dt1;
			p1.y = y_a_coefficient[i] + y_b_coefficient[i]*dt1 + y_c_coefficient[i]*dt1*dt1 + y_d_coefficient[i]*dt1*dt1*dt1;
			p2.x = x_a_coefficient[i] + x_b_coefficient[i]*dt2 + x_c_coefficient[i]*dt2*dt2 + x_d_coefficient[i]*dt2*dt2*dt2;
			p2.y = y_a_coefficient[i] + y_b_coefficient[i]*dt2 + y_c_coefficient[i]*dt2*dt2 + y_d_coefficient[i]*dt2*dt2*dt2;
			// circle(tmp_img1, Point(p1.x, p1.y), 1, Scalar(255, 0, 0), -1);
			line(tmp_img1, Point(p1.x, p1.y), Point(p2.x, p2.y), Scalar(255, 0, 0), 3);
		}
	}

	for(int i = 0; i< fixed_path.size(); ++i){
		circle(tmp_img1, Point(mix_vex_edge[fixed_path[i] - 1].x, mix_vex_edge[fixed_path[i] - 1].y), 4, Scalar(0, 0, 255), -1);
		// cout << Point(mix_vex_edge[fixed_path[i] - 1].x, mix_vex_edge[fixed_path[i] - 1].y) << endl; 
	}
	myNameSpace::myImshow("test", tmp_img1);
}
*/
Voronoi_vertex::Voronoi_vertex(){

}

Voronoi_vertex::~Voronoi_vertex(){

}