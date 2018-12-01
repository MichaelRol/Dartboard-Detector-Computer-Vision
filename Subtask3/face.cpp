/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;
/** Function Headers */
void detectAndDisplay( Mat frame , string filename);
bool checkMatch(string topleftx, string toplefty, string bottomrightx, string bottomrighty, std::vector<Rect> face);
void calcF1(std::vector<Rect> faces, string csv);
/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv ) {
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	//detectAndDisplay( frame , argv[1]);

	Mat grad;
	Mat frame_gray;

	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	int c;

	/// Load an image

	GaussianBlur( frame, frame, Size(3,3), 0, 0, BORDER_DEFAULT );

	/// Convert it to gray
	cvtColor( frame, frame_gray, CV_BGR2GRAY );

	/// Create window

	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( frame_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( frame_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );

  //get the magnitude
	// Mat magnitude
	// Mat angle
	// cartToPolar(grad_x,grad_y,magnitude,angle,0);
	// 

	Mat dest;
	threshold(grad, dest, 100, 0, 3);

	// 4. Save Result Image
	imwrite( "output1.jpg", dest );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame , string filename) {
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ ) {
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
  calcF1(faces, filename);

}

bool checkMatch(string topleftx, string toplefty, string bottomrightx, string bottomrighty, Rect		 face) {
  if (abs(stoi(topleftx) - face.x) < 100 && abs(stoi(toplefty) - face.y) < 100 \
      && abs(stoi(bottomrightx) - (face.x + face.width)) < 100 && abs(stoi(bottomrighty) - (face.y + face.height)) < 100) {
		return true;
	} else {
	 	return false;
	}
}

void calcF1(std::vector<Rect> faces, string csv) {

	string subname = csv.substr(14, csv.size() - 18);
	string csvname = "CSVs/boardcoords" + subname + ".csv";

	ifstream ip(csvname);

	if(!ip.is_open()) std::cout << "Error; File Open" << '\n';

	string topleftx;
	string toplefty;
	string bottomrightx;
	string bottomrighty;

	double numfaces = 0;
	double truepositives = 0;
  double falsenegatives = 0;
	double falsepositives = 0;

	while(ip.good()) {
		getline(ip, topleftx, ',');
		getline(ip, toplefty, ',');
		getline(ip, bottomrightx, ',');
		getline(ip, bottomrighty, '\n');

		if (ip.eof()) break;
		numfaces++;

		for (int i = 0; i < faces.size(); i++) {
			if (checkMatch(topleftx, toplefty, bottomrightx, bottomrighty, faces[i])) {
				truepositives++;
				break;
			}
		}
	}
  falsepositives = faces.size() - truepositives;
	falsenegatives = numfaces - truepositives;

	double f1score = 2*truepositives/(2*truepositives + falsenegatives + falsepositives);

	cout << "True number of faces: " << numfaces << endl << "True Positives: " \
	          << truepositives << endl << "TPR: " << truepositives/numfaces << endl \
						<< "F1 Score: " << f1score << endl;

	ip.close();

}
