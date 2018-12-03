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
Mat generateLineHoughSpace(Mat gradMag, Mat gradDir);
Mat generateCircleHoughSpace(Mat gradMag, Mat gradDir);
Mat detectBoards(Mat originalImage, Mat houghSpace, int* buckets);
Mat drawCircles (Mat originalImage, Mat houghSpace);
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

    Mat houghSpace = generateLineHoughSpace(gradMag, gradDir);
	int buckets[20*20] = {0};
	Mat detected = detectBoards(frame, houghSpace, buckets);

	// 4. Save Result Image
	string filename = argv[1];
	string outputname = filename.substr(10, filename.size() - 14);
	//imwrite( "Detected/"+outputname+".jpg", output );

	imwrite( "Detected/"+outputname+".jpg", detected );

	return 0;
}

Mat detectBoards(Mat originalImage, Mat houghSpace, int* buckets) {

	double pi = 3.1415926535897;
  int width = originalImage.size().width;
	int height = originalImage.size().height;
  int count = 0;
	int bucketavgx[20*20] = {0};
	int bucketavgy[20*20] = {0};
	int bucketsizex = floor(width/20);
	int bucketsizey = floor(height/20);

	for (int degrees = 0; degrees < houghSpace.size().width; degrees++) {
		for (int rho = 0; rho < houghSpace.size().height; rho++) {
			if (houghSpace.at<int>(rho, degrees) > 170){
				count++;
				// int crossx = (rho-width-height)/cos(degrees*pi/180);
				// int crossy = (rho-width-height)/sin(degrees*pi/180);
				// int crosswidth = ((rho-width-height)-width*cos(degrees*pi/180))/sin(degrees*pi/180);
				// int crossheight = ((rho-width-height)-height*sin(degrees*pi/180))/cos(degrees*pi/180);
				// line(originalImage, Point(0, crossy), Point(width, crosswidth), Scalar( 0, 255, 0 ), 1);
			}
		}
	}
	
	int linearray[count*2];
	int counted = 0;
	for (int degrees = 0; degrees < houghSpace.size().width; degrees++) {
		for (int rho = 0; rho < houghSpace.size().height; rho++) {
			if (houghSpace.at<int>(rho, degrees) > 170){
				linearray[counted*2] = rho - width - height;
				linearray[counted*2+1] = degrees;
				counted++;
			}
		}
	}

	for (int line = 0; line < count; line++) {
		for (int restoflines = 0; restoflines < count; restoflines++) {
		  if (line != restoflines) {

				int degrees0 = linearray[line*2+1];
				int degrees1 = linearray[restoflines*2+1];

				int r0 = linearray[line*2];
				int r1 = linearray[restoflines*2];
				float a1 = cos((float)degrees1*pi/180);
				float b0 = sin((float)degrees0*pi/180);
				float a0 = cos((float)degrees0*pi/180);
				float b1 = sin((float)degrees1*pi/180);

				int x = round((r0*b1 - b0*r1)/(a0*b1 - b0*a1));

				int y = round((a0*r1 - r0*a1)/(a0*b1 - b0*a1));

				for(int bucketx = 0; bucketx < 20; bucketx++) {
					for (int buckety = 0; buckety < 20; buckety++) {
						if (x >= bucketx * bucketsizex && x < (bucketx+1)*bucketsizex && y >= buckety * bucketsizey && y < (buckety+1)*bucketsizey) {
							buckets[bucketx*20+buckety]++;
							bucketavgx[bucketx*20+buckety] = (bucketavgx[bucketx*20+buckety]*(buckets[bucketx*20+buckety]-1) + x)/buckets[bucketx*20+buckety];
							bucketavgy[bucketx*20+buckety] = (bucketavgy[bucketx*20+buckety]*(buckets[bucketx*20+buckety]-1) + y)/buckets[bucketx*20+buckety];

						}
					}
				}
			}
		}
	}

		// for (int x = 0; x < 20; x++) {
		// 	for (int y = 0; y < 20; y++) {
		// 		if (buckets[x*20+y] > 20) {
		// 			cout << "X: " << bucketavgx[x*20+y] << " Y: " << bucketavgy[x*20+y] << endl;
		// 		}
		// 	}
		// }

		std::vector<Rect> boards;
		Mat frame_gray;

		// 1. Prepare Image by turning it into Grayscale and normalising lighting
		cvtColor( originalImage, frame_gray, CV_BGR2GRAY );
		equalizeHist( frame_gray, frame_gray );

		// 2. Perform Viola-Jones Object Detection
		cascade.detectMultiScale( frame_gray, boards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );
		// for( int i = 0; i < boards.size(); i++ ) {
	 	// 	rectangle(originalImage, Point(boards[i].x, boards[i].y), Point(boards[i].x + boards[i].width, boards[i].y + boards[i].height), Scalar( 0, 255, 0 ), 2);
    // }
		// 4. Draw box around faces found

		for(int x = 0; x < 20; x++){
			for(int y = 0; y < 20; y++){
				for( int i = 0; i < boards.size(); i++ ) {
					int topleftx = boards[i].x;
					int toplefty = boards[i].y;
					int bottomrightx = boards[i].x + boards[i].width;
					int bottomrighty = boards[i].y + boards[i].height;

					if(bucketavgx[x*20+y] > topleftx && bucketavgx[x*20+y] < topleftx + bottomrightx \
					&& bucketavgy[x*20+y] > toplefty && bucketavgy[x*20+y] < toplefty + bottomrighty) {
						rectangle(originalImage, Point(boards[i].x, boards[i].y), Point(boards[i].x + boards[i].width, boards[i].y + boards[i].height), Scalar( 0, 255, 0 ), 2);
					}
				}
			}
		}

		//std::cout << boards.size() << std::endl;

		//calcF1(boards, filename);

	//line(originalImage, Point(width, crosswidth), Point(crossheight, height), Scalar( 0, 255, 0 ), 1);
	//line(originalImage, Point(0, crossy), Point(crossx, 0), Scalar( 0, 255, 0 ), 1);
	// line(originalImage, Point(0, crossy), Point(crossheight, height), Scalar( 0, 255, 0 ), 1);
	// line(originalImage, Point(crossx, 0), Point(crossheight, height), Scalar( 0, 255, 0 ), 1);
	// line(originalImage, Point(crossx, 0), Point(width, crosswidth), Scalar( 0, 255, 0 ), 1);
	return originalImage;
}

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

Mat generateLineHoughSpace(Mat gradMag, Mat gradDir) {


	int width = gradMag.size().width;
	int height = gradMag.size().height;
	Mat houghSpace(2*(height+width), 360, CV_32SC1, Scalar(0));
	Mat output;
	double pi = 3.1415926535897;

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++ ) {

			if (gradMag.at<uchar>(y, x) != 0) {

				for (int deg = 0; deg < 360; deg++) {
					int rho = round(x*cos(deg*pi/180) + y*sin(deg*pi/180) + width + height);
					houghSpace.at<int>(rho, deg)++;
				}

			}

		}
	}
  normalize(houghSpace, output, 0, 255, NORM_MINMAX, CV_32SC1);
	imwrite( "hough.jpg", output );
  return output;
}

Mat prepImage(Mat frame) {

	  Mat frame_gray;
	  Mat holder;
		GaussianBlur( frame, holder, Size(3,3), 0, 0, BORDER_DEFAULT );
		cvtColor( holder, frame_gray, CV_BGR2GRAY );

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
		threshold(gradMag, dest, 150, 255, 0);

		return dest;
}

/** @function detectAndDisplay */

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
