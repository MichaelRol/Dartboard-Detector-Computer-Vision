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
#include <math.h>

using namespace std;
using namespace cv;
/** Function Headers */
void detectAndDisplay( Mat frame , string filename);
bool checkMatch(string topleftx, string toplefty, string bottomrightx, string bottomrighty, std::vector<Rect> face);
void calcF1(std::vector<Rect> faces, string csv);
Mat getGradMag(Mat frame_gray);
Mat getGradDir(Mat frame_gray);
Mat prepImage(Mat frame);
Mat generateHoughSpace(Mat gradMag, Mat gradDir);
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

  Mat prepedImage = prepImage(frame);
	Mat gradMag = getGradMag(prepedImage);
	Mat gradDir = getGradDir(prepedImage);

  Mat houghSpace = generateHoughSpace(gradMag, gradDir);

	// 4. Save Result Image
	imwrite( "outputMag.jpg", gradMag );
	imwrite( "outputDir.jpg", gradDir );

	return 0;
}

Mat generateHoughSpace(Mat gradMag, Mat gradDir) {


	int width = gradMag.size().width;
	int height = gradMag.size().height;
	Mat houghSpace(2*(height+width), 360, CV_32SC1, Scalar(0));
	double pi = 3.1415926535897;

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++ ) {

			if (gradMag.at<uchar>(y, x) != 0) {

				float theta = gradDir.at<float>(y, x) + pi;
				int rho = round((x*cos(theta) + y*sin(theta)) + width + height);


        int degrees = round(theta * (180/pi));

				cout << rho << endl;
			  houghSpace.at<int>(rho, degrees) += 20;

			}

		}
	}

	imwrite( "hough.jpg", houghSpace );
  return houghSpace;
}

Mat prepImage(Mat frame) {

	  Mat frame_gray;
		GaussianBlur( frame, frame, Size(3,3), 0, 0, BORDER_DEFAULT );
		cvtColor( frame, frame_gray, CV_BGR2GRAY );

		return frame_gray;
}

Mat getGradDir(Mat frame_gray) {
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;

	int c;
  double pi = 3.1415926535897;
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( frame_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	//convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( frame_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	//convertScaleAbs( grad_y, abs_grad_y );

	Mat gradDir(frame_gray.size().height, frame_gray.size().width, CV_32F);

	for (int x = 0; x < frame_gray.size().width; x++) {
		for (int y = 0; y < frame_gray.size().height; y++) {
			if (grad_y.at<float>(y, x) != 0) {
				gradDir.at<float>(y, x) = atan2(grad_x.at<float>(y, x),grad_y.at<float>(y, x));
			} else {
				gradDir.at<float>(y, x) = pi/2;
			}
		}
	}

	return gradDir;
}

Mat getGradMag(Mat frame_gray) {

		int scale = 1;
		int delta = 0;
		int ddepth = CV_16S;

		int c;

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
		Mat gradMag;
		addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradMag );

		Mat dest;
		threshold(gradMag, dest, 100, 255, 0);

		return dest;
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
