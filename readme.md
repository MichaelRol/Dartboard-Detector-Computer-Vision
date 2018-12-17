Dartboard Detector
==================
**Created for UoB Image Processing and Computer Vision course**

Compile with appropriate OpenCV libraries linked. 
You can then run the outputed executable with one arguement which is the path to the image you wish to run the detector on. 
Ensure cascade.xml is in the same directory as the executable.
Example images are given in the Originals/ folder. The outputed image will be the same as the input except with green rectangles drawn around what is detected as a dartboard. Examples of the magnitude gradient and Hough space images are also included in the repo.

**To obtain performace statistics, including F1 score**

* Uncomment call to calcF1() in the detectBoards() function.
* Recompile
* In the same directory as the executable and and jpeg image create a CSV file named darts.csv
* In the CSV file type coordinates for the top right and bottom left corners of 'imaginary' rectangles which contain true dartboard images. The pattern should be x,y,x,y of the top right followed by bottom left.

