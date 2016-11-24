/**
* @file   util.h
* @Author Tick Son Wang, EE Heng Chen
* @date   05.07.2016
* @brief  Header file for the utility functions
*
* Detailed description of file.
*/



/******************************************************************************
Dependencies
******************************************************************************/
//#include <opencv2\opencv.hpp> // change to ind. modules for min. footprint
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <stack>
#include <ctime>

#include <XVImageSeq.h>
#include <XVMpeg.h>
#include <XVImageIO.h>
#include <XVColorSeg.h>
#include <XVBlobFeature.h>
#include <XVTracker.h>
#include <XVWindowX.h>
#include "ippi.h"


#include "aruco/markerdetector.cpp"
#include "aruco/marker.cpp"

#include "aruco/ar_omp.cpp"
#include "aruco/subpixelcorner.cpp"
#include "aruco/cameraparameters.cpp"
#include "aruco/arucofidmarkers.cpp"


std::stack<clock_t> tictoc_stack;


/******************************************************************************
Macros & Macro Functions
******************************************************************************/
template < typename T > std::string to_string( const T& n )
{
	std::ostringstream stm ;
	stm.width(4); stm.fill('0');
	stm << n ;
	return stm.str() ;
}



/******************************************************************************
Structures
******************************************************************************/
/*! @struct MouseCallBackHistData

@brief A structure to pass histogram data to the opencv MouseCallBack function

*/
struct MouseCallBackHistData{
	int _hscale, //!< hue scale
		_sscale; //!< saturation scale
	int _hbin_scale, //!< hue bin scale
		_sbin_scale; //!< saturation bin scale
	cv::Mat	_hist; //!< contains the hist data
	int _h_mean; //!< mean hue
	int _s_mean; //!< mean saturation
};

/*! @struct MouseCallBackHistData

@brief A structure to pass histogram data to the opencv TrackbarCallBack function

*/
struct TrackbarCBDataCalib{
	cv::Mat _hsv; //!< hsv data
	cv::Mat _seg_mask; //!< segmentation mask {0,255}
	cv::Mat _seg_out_rgb;  //!< segmented image in rgb
	cv::Mat _seg_in_rgb; //!< src_image for segmentation in rgb
	int 	_H_top, //!< upper bound for hue
			_H_bot, //!< lower bound for hue
			_S_top, //!< upper bound for saturation
			_S_bot; //!< lower bound for saturation
	std::string _wn_seg_mask; //!< window name for segmentation mask
	std::string _wn_seg_rgb; //!< window name for segmentated rgb image
};


/*! @struct DatasetPath

@brief a structure to ease dataset pathing

@param  _fd folder directory
@param _sfd subfolder directory
@param _fpref file prefix
@param _fpstf file postfix
@param _dd directory delimiter
@param _s_idx start index
@param _e_idx end index
@param _c_idx current index
*/
struct DatasetPath{
	std::string _fd; //!< folder directory
	std::string _sfd; //!< subfolder directory
	std::string _fpref; //!< file prefix
	std::string _fpstf; //!< file postfix
	std::string _dd; //!< directory delimiter 
	int _s_idx; //!< start index 
	int _e_idx; //!< end index
	int _c_idx; //!< current index

	/*! @fn Parameterized Constructor

	@brief Parameterized constructor
	@param model_file directory to the deploy.prototxt
	@param trained_file directory to the model.caffemodel
	@param label_file directory to the text file (synset_words.txt) containing the name of the classes.
	
	*/
	DatasetPath(std::string fd,
				std::string fpref,
				std::string fpstf,
				int s_idx, 
				int e_idx,
				std::string sfd ="/",
				std::string dd ="/" ){

		_fd= fd; 
		_fpref = fpref;
		_fpstf = fpstf;
		_s_idx = s_idx;
		_e_idx = e_idx;
		_c_idx = s_idx;
		_sfd = sfd;
		_dd = dd;
	}


	std::string path(int idx);
	std::string path_4digits(int idx);
};

/*! @struct DatasetPath

*
*/
struct DenseScanFrames{
	int _img_w;
	int _img_h;
	int _class_int;
	cv::Point2i _pos;
	cv::Mat _sub_img;
};
/******************************************************************************
Classes
******************************************************************************/

/*! @class Dataset

	@brief A class which provides the functionality to read all image files in a folder

*/

class Dataset{
	
public:
	std::vector<cv::Mat> _img; //!< images
	std::vector<std::string> _fn; //!< corresponding images file names
	std::vector<int> _label; //!< corresponding labels
	int _size; //!< number of read images
	std::string _dir; //!< directory of dataset

	/*! @fn
	 *  @brief Default constructor
	 */
	Dataset(){};

	/*! @fn
	 *  @brief Parameterized constructor to read all the images in a folder.
	 *  @param dir folder, under which the images are read.
	 *	@param read_label set to true to read the labels off the image file name
	 *	@param read_img set to true to read and store image data 
	 */
	Dataset(std::string dir,
			bool read_label = true,
			bool read_img = true,
			std::string img_extension=".jpg");
};

/******************************************************************************
Function Headers
******************************************************************************/

cv::Vec3f movingAveragePoint(std::vector<cv::Vec3f>points,float window);

float movingAverageFloat(std::vector<float> data,float window);

double movingAverageDouble(std::vector<double> data,float window);

cv::Vec3f pointToVelocity(cv::Vec3f p1, cv::Vec3f p2);

float pointToSpeed(cv::Vec3f p1, cv::Vec3f p2);

void segmentHSVEDIT(cv::Mat src_hsv, cv::Mat& seg_mask,
                    int h_top, int h_bot, int s_top, int s_bot);

int markerContact(std::vector<cv::Vec3f> marker_center, cv::Vec4f plane, cv::Vec3f object_point);

void depthImaging(cv::Mat &depth_image, cv::Mat depth_global, uint16_t* mGamma);

void pointCloudTrajectory(cv::Mat cloud, cv::Vec3f &point, std::vector<double> &EV);

void normalPlaneCheck(cv::Vec4f &plane_equation);

std::vector<aruco::Marker> arucoMarkerDetector(cv::Mat &rgb, bool write_id, bool display_id);

cv::Vec4f RANSAC3DPlane(cv::Mat cloud, cv::Mat &plane, int iter, float *ratio, float threshold);

cv::Vec3f computePlane(cv::Vec3f A, cv::Vec3f B, cv::Vec3f C);

cv::Vec3f normalization3D(cv::Vec3f vec);

cv::Vec3f crossProd(cv::Vec3f A, cv::Vec3f B);

float dotProd(cv::Vec3f A, cv::Vec3f B);

std::vector<cv::Rect> detectFaceAndEyes( cv::Mat frame , cv::CascadeClassifier face_cascade);

void tic();

void toc();

bool contactCheck(cv::Mat hand, cv::Rect object_blob);

void noiseRemove(cv::Mat seg_mask, cv::Mat& seg_mask_noisefree, cv::Rect& box);

void noiseRemoveBox(cv::Mat seg_mask, cv::Mat& seg_mask_noisefree, cv::Rect& box);


void mouseCallBackHist(	int eventcode, 
						int x, int y, 
						int flags, 
						void* data);


void trackbarCallBackCalib(int trackpos, void* data);

/*! @fn void getColorThreshold(cv::Mat src,
                               int(&hue_range)[2], int(&sat_range)[2])
@brief gets the value of hue and sat
@param src input image, assert [CV_8UC1]
@param hue_range range of hue values in the input image
@param sat_range range of saturation values in the input image

From the src image, a color is selected manually and the range of hue and sat
for that color is saved.
*/
void getColorThreshold(cv::Mat src, int(&hue_range)[2], int(&sat_range)[2]);


/*! @fn cv::Mat imgCrop(cv::Mat img_src, cv::Point2i cent,
				        cv::Size2i size = cv::Size(32, 32));
@brief crops an image based on center point and size
@param img_src input image
@param cent center point of crop
@param size size of crop (width,height)
*/
cv::Mat imgCrop(cv::Mat img_src, cv::Point2i cent,
				cv::Size2i size = cv::Size(32, 32));

/*! @fn void segmentHSV(cv::Mat src_hsv, cv::Mat src_rgb,
				cv::Mat& seg_mask, cv::Mat& seg_rgb,
				int h_top, int h_bot, int s_top, int s_bot)
@warning rang(hue)=[0,179], range(saturation)=[0,255]
@brief Segment an image given the hue and saturation range
@param src_hsv hsv input image
@param src_rgb rgb input image
@param seg_mask output segmentation binary mask
@param seg_rgb output segmentation image in rgb
@param h_top upper threshold for hue
@param h_bot lower threshold for hue
@param s_top upper threshold for saturation
@param s_bot lower threshold for saturation

*/
void segmentHSV(cv::Mat src_hsv, cv::Mat src_rgb,
				cv::Mat& seg_mask, cv::Mat& seg_rgb,
				int h_top, int h_bot, int s_top, int s_bot);


/*! @fn std::vector<DenseScanFrames> denseScan(cv::Mat src, int o_x, int o_y,
							float up_scale, int level,
							int w, int h)
@brief Returns all the dense scanimage windows according to the specified 
parameters.
@param src input image 
@param o_x x-offset per scan window
@param o_y y-offset per scan window
@param up_scale factor to scale up window in the next run 
@param level number of scale level
@param w scan window width
@param h scan window height
*/
std::vector<DenseScanFrames> denseScan(cv::Mat src, int o_x, int o_y,
							float up_scale, int level,
							int w, int h);


void trackbarCallBackCalib_H_top(int trackpos, void* data);
void trackbarCallBackCalib_H_bot(int trackpos, void* data);
void trackbarCallBackCalib_S_top(int trackpos, void* data);
void trackbarCallBackCalib_S_bot(int trackpos, void* data);

/*! @fn bool fexists(const char *filename);
@brief checks if file exists.
@param filename is the name of the file.
@return gives a non-zero value if file exists.
*/
bool fexists(const char *filename);

/*! @fn bool fexists(const std::string& filename)
@brief checks if file exists.
@param filename is the name of the file.
@return gives a non-zero value if file exists.
*/
bool fexists(const std::string& filename);

/*! @fn cv::Mat imgResize(cv::Mat src, int width, int height)
@brief resizes an image
@param src is path name to the source file.
@param width is the new width the image.
@param height is the new height of the image.
@return creates a resized image.
*/
cv::Mat imgResize(cv::Mat src, int width, int height);

/*! @fn bool copyFile(const char *SRC, const char* DEST)
@brief copies a file from SRC to DEST.
@param SRC is path name to the source file.
@param DEST is the path name to the intended destination file.
@return gives a non-zero value for a successful copy.
*/
bool copyFile(const char *SRC, const char* DEST);

/*! @fn void createCaffeDataList(const char** DATAFOLDER, int* CONFIG)
@warning Images in DATAFOLDER have to be cropped first.
@brief groups images and creates a txt file for Caffe.
@param DATAFOLDER is the folder name/s with the image datasets.
@param CONFIG is the configuration parameters.
@return groups images and creates a txt file for Caffe.

This function groups the cropped images in DATAFOLDER to a single
test/train folder and creates a text file that includes the name
of the cropped images and the class label for Caffe.
*/
void createCaffeDataList(const char** DATAFOLDER, int* CONFIG);

/*! @fn void createCaffeDataList(const char* path, const char* list)
@brief creates a txt file for Caffe.
@param path is the path to the Caffe folder.
@param list is the name of the text file.
@return creates a txt file for Caffe.

This function reads from the path the name of the files and creates
a list that includes the name of the files and the class label.
*/
void createCaffeDataList(const char* path, const char* list);

/*! @fn void groupTrainTestFile(char* src_path, char* caffe_path, int* NUM, int(&counter))
@brief groups the files into train/test folder for Caffe.
@param src_path is the path to the original images.
@param caffe_path is the path to the folder created for Caffe.
@return creates a folder for Caffe with cropped train/test images.

This function copies the images from the folders containing the
cropped images that is needed for training/testing into a single
test/train folder that is used for Caffe. Here the number of images copied
can be specified in CONFIG.
*/
void groupTrainTestFile(char* src_path, char* caffe_path, int* NUM, int(&counter));

/*! @fn int groupDataSetFile(char* rec_path, const char* rec_path_all,
                             int* CONFIG)
@brief groups the dataset inside the DATAFOLDER.
@param rec_path is the path to DATAFOLDER.
@param rec_path_all is the folder path where the datasets are grouped.
@param CONFIG is the configuration parameters.
@return creates a folder with all datasets and returns the dataset number.

This function groups all the datasets in DATAFOLDER into a single folder
and counts the number of datasets.
*/
int groupDataSetFile(char* rec_path, const char* rec_path_all, int* CONFIG);

/*! @fn serialCropImg(const char** DATAFOLDER, char* train_test, int* CONFIG)
@brief crops images for each individual classes.
@param DATAFOLDER is the folder name/s with the image datasets.
@param train_test is the folder name with DATAFOLDER.
@param CONFIG is the configuration parameters.
@return creates a folder inside DATAFOLDER with cropped images.

The dataset should be organised as follows manually:\n
train_test\n
~ DATAFOLDER\n
~~ dataset\n
\n

This function reads through all the DATAFOLDER. For each DATAFOLDER, it opens
a sample image from dataset and allows the user to manually chose the color
of the object in HSV space. Segmentation mask is created based on it. The HSV
values are saved for future use. The segmentation mask is then used to
crop the objects from the source image. There is an option to resize the
cropped images before saving.
*/
void serialCropImg(const char** DATAFOLDER, char* train_test, int* CONFIG);

/*! @fn std::vector<cv::Mat> cropImgs(cv::Mat seg_mask, cv::Mat src,
                                  int* CONFIG)
@brief crops image based on the segmentation mask.
@param seg_mask is the path to the folder with the images
@param src is the source image.
@param CONFIG is the configuration parameters.
@return creates an array of cropped images.

This function crops a series of images from a source image based on
the segmentation mask defined. The cropping is done by finding the biggest
bounding box that encloses a single object segmented by the segmentation
box. The bounding box is shifted [nxm - 1] times following a grid pattern,
resulting in [nxm] boxes. The source image is cropped according to the
bounding boxes, resulting in nxm number of cropped images. The bounding
boxes are enlarged to include some background.
*/
std::vector<cv::Mat> cropImgs(cv::Mat seg_mask, cv::Mat src, int* CONFIG);

/*! @fn cv::Mat segmentation(cv::Mat src, int* hue_range, int* sat_range)
@brief creates the segmentation mask.
@param src is the source image.
@param hue_range is the range of the hue value of a color.
@param sat_range is the range of the sat value of a color.
@return creates a segmentation mask of [0,1]

This function creates the segmentation mask of a color based on hue_range
and sat_range. The resulting mask has a range of [0,1] and has the same
size as the sorce image.
*/
cv::Mat segmentation(cv::Mat src, int* hue_range, int* sat_range);

/*! @fn void serialCropNegImg(const char** DATAFOLDER, char* train_test,
                          int* CONFIG)
@brief crops images for the negative class.
@param DATAFOLDER is the folder name/s with the image datasets.
@param train_test is the folder name with DATAFOLDER.
@param CONFIG is the configuration parameters.
@return creates a folder inside DATAFOLDER with cropped images.

This function crops the images in DATAFOLDER for the negative class.
It carries out segmentation on the non-negative class specified in CONFIG
and uses the position of the resulting bounding box to crop the images
in DATAFOLDER for the negative class.
[WARNING] Use this fcn after the non-negative class has been cropped.
*/
void serialCropNegImg(const char** DATAFOLDER, char* train_test,
		                 int* CONFIG);

/*! @fn void imgVar(char* dst_path)
@brief calculates the variance of a set of images.
@param src_path is the path to the folder with the images.

This function calculates the variance over a set of images and shows the
resulting variance as an image. The function can be modified to save the
variance as an image file.
*/
void imgVar(char* src_path);


//[MAIN PAGE DOC]*************************************************************

/*! \mainpage The mainpage documentation

This is the documentation of the C++ codes used in this project.

A general introduction to the files in the codes

1. <util.h, util.cpp>

	The <util> library contain utility functions used in the several main programs
	in the projects.

2. <caffe_classifier.h, caffe_classifier.cpp>

	The <caffe_classifier> library contains the classifier class to invoke a trained
	Caffe model and make prediction onto an input image.
	An example program using this classifier class is the <main_valid_eval.cpp>.

3. <main_valid_eval.cpp>  (Requires proper caffe installation and eclipse setup)

	This is an example program which uses the classifier class to find the global
	and individual threshold on the prediction confidence level in order to filter
	out non-discriminative predictions.

4. <main_crop.cpp>.

	This is an example program for automated training data generation.
	Configurations have to be done first.

*/




