/*
 * main.cpp
 *
 *  Created on: Dec 6, 2010
 *      Author: papazov
 */

#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <signal.h>
#include <list>
#include <iostream>
#include <math.h>
#include <vector>
#include <iostream>
#include <semaphore.h>

#include "util.cpp"

#include <XVImageSeq.h>
#include <XVMpeg.h>
#include <XVImageIO.h>
#include <XVColorSeg.h>
#include <XVBlobFeature.h>
#include <XVTracker.h>
#include <XVWindowX.h>
#include "ippi.h"
#include "ippcv.h"
#include "ippcc.h"

using namespace std;


// object detetctor
const int color_hand = 40, color_obj = 358, range_limit = 10;
inline bool check_hand(const u_short value)
{return (value-color_hand<13  && color_hand-value<13);}
inline bool check_obj(const u_short value)
{return (value-color_obj<5  && color_obj-value<5);}

// image variables
cv::Mat rgb_global = cv::Mat::zeros(480,640,CV_8UC3);
cv::Mat depth_global,cloud_global;
cv::Mat mask_head_global = cv::Mat::zeros(480,640,CV_8UC1);
cv::Mat mask_hand_global = cv::Mat::zeros(480,640,CV_8UC1);
cv::Mat mask_obj_global  = cv::Mat::zeros(480,640,CV_8UC1);
std::vector<cv::Vec3f> marker_center_global;
cv::Vec4f plane_global;
cv::Rect object_blob_global;
float frame_number_global = 0.0;
bool contact_obj = false;
int  contact_marker = 0;
cv::Vec3f single_point_obj_global;

cv::Vec3f faces_global;

std::vector<double> eig_val_global(3);

// threads
int MAX = 4;
sem_t mutex_t1,mutex_t5,mutex_t6;
sem_t lock_t1,lock_t2,lock_t3,lock_t4,lock_t5,lock_t6;

bool flag_thres = true;
bool flag_plane = false;
bool flag_marker = false; int marker_num = 2;

// option flags
//#define FLAG_RGB
//#define FLAG_DEPTH
//#define FLAG_MARKER
//#define FLAG_PLANE
#define FLAG_OBJECT
#define FLAG_HAND
//#define FLAG_FACE
#define FLAG_THREAD
#define FLAG_WRITE





//====================================================================================================================================
// [IPPI WINDOW]***************************************************************
void openWindow_8u(Ipp8u *img, IppiSize *size, int nChannels, char *name)
{
  IplImage *cvImg;
  CvSize sizeCv;
  Ipp8u tmp[640*480*3] = {0};
  sizeCv.width = size->width;
  sizeCv.height = size->height;
  cvImg = cvCreateImage(sizeCv,IPL_DEPTH_8U,nChannels);
  ippiCopy_8u_C3R(img,size->width*nChannels,tmp,size->width*nChannels,*size);
  cvSetData(cvImg,(void*) tmp,sizeCv.width*nChannels);
  cvNamedWindow(name,1 ); 
  cvShowImage(name,cvImg); 
}

//====================================================================================================================================
// [THREAD 1 : KINECT]*********************************************************
void* kinectGrab(void* v_kinect){

  cv::VideoCapture * kinect = reinterpret_cast<cv::VideoCapture *>(v_kinect);

  // Depth value processing
  uint16_t mGamma[2048];
  for( int i=0;i<2048;++i )
    { float v=i/2048.0; v=powf(v, 3)*6; mGamma[i]=v*6*256;}   

  // Initialize images
  cv::Mat depth_image = cv::Mat::zeros(480,640,CV_8UC3);
  cv::Mat plane_tmp   = cv::Mat::zeros(480,640,CV_8UC1);
  cv::Mat rgb_marker  = cv::Mat::zeros(480,640,CV_8UC3);

  // Variables
  std::vector<aruco::Marker> marker;
  float ratio[2]; ratio[0] = 0.2; ratio[1] = 0.5;
  char keypress;

  while(true){
    sem_wait(&lock_t1);
    sem_wait(&lock_t1);
    //sem_wait(&lock_t1);
    sem_wait(&lock_t1);
    sem_wait(&lock_t1);
    sem_wait(&mutex_t1);

    kinect->grab();
    kinect->retrieve(rgb_global,CV_CAP_OPENNI_BGR_IMAGE);
    kinect->retrieve(depth_global,CV_CAP_OPENNI_DEPTH_MAP);
    kinect->retrieve(cloud_global,CV_CAP_OPENNI_POINT_CLOUD_MAP);
    frame_number_global =
      kinect->get(CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_POS_FRAMES);
    printf("FRAME : %f     ",frame_number_global);

#ifdef FLAG_DEPTH
    depthImaging(depth_image,depth_global,mGamma);
    cv::imshow("depth",depth_image); cvWaitKey(1);
#endif

#ifdef FLAG_RGB
    cv::imshow("rgb",rgb_global); cvWaitKey(1);
#endif

    while(!flag_plane){          
      plane_tmp = cv::Mat::zeros(480,640,CV_8UC1);
      plane_global = RANSAC3DPlane(cloud_global, plane_tmp, 500, ratio, 0.005);
      normalPlaneCheck(plane_global);
#ifdef FLAG_PLANE 
      cv::imshow("plane",plane_tmp*255);
      printf("SAVE NORMAL VECTOR OF PLANE : [Y/N] \n\n");
      keypress = cv::waitKey(0); 
      if (keypress == 'y') {flag_plane = true; cv::destroyWindow("plane");} 
      cv::waitKey(30);
#else
      flag_plane = true;
#endif
    }

    if(!flag_marker){
      marker_center_global.clear();
      marker.clear();
      rgb_marker = cv::Mat::zeros(480,640,CV_8UC3);
      rgb_marker = rgb_global.clone();
      marker = arucoMarkerDetector(rgb_marker, false, true); //write_id, display_id
      if(marker.size()==2){
        for(int i=0;i<marker.size();i++)
          marker_center_global.push_back(cloud_global.at<cv::Vec3f>
                                        (marker[i].getCenter().y,
                                         marker[i].getCenter().x));
      }
#ifdef FLAG_MARKER 
      cv::imshow("rgb_m",rgb_marker); 
      printf("\nSAVE MARKERS : [Y/N] \n\n");
      keypress = cv::waitKey(0); 
      if (keypress == 'y') {flag_marker = true; cv::destroyWindow("rgb_m");}
      cv::waitKey(30);
#else
      if(marker.size()==2) flag_marker = true;
#endif
    }

    sem_post(&mutex_t1);
    sem_post(&lock_t2);
    sem_post(&lock_t3);
    //sem_post(&lock_t4);
    sem_post(&lock_t5);
    sem_post(&lock_t6);
  }
  return 0;
}

//====================================================================================================================================
// [THREAD 2 : OBJECT DETECTOR]************************************************
void* objectDetector(void* arg)
{

  int i = 0;
  int found_index = 0;
  float found_size = 0.0;

  int hue_range_obj[2], sat_range_obj[2];
//  hue_range_obj[0] = 115; hue_range_obj[1] = 132;
//  sat_range_obj[0] = 133; sat_range_obj[1] = 255;

//red bar
  hue_range_obj[0] = 116; hue_range_obj[1] = 138;
  sat_range_obj[0] = 199; sat_range_obj[1] = 255;

// blue board
//  hue_range_obj[0] = 0; hue_range_obj[1] = 81;
//  sat_range_obj[0] = 110; sat_range_obj[1] = 168;

// light green cup
//  hue_range_obj[0] = 66; hue_range_obj[1] = 93;
//  sat_range_obj[0] = 31; sat_range_obj[1] = 77;

// green cup
//  hue_range_obj[0] = 77; hue_range_obj[1] = 98;
//  sat_range_obj[0] = 76; sat_range_obj[1] = 214;

// yellow plyers
//  hue_range_obj[0] = 80; hue_range_obj[1] = 102;
//  sat_range_obj[0] = 135; sat_range_obj[1] = 255;

// yellow Banana
//  hue_range_obj[0] = 72; hue_range_obj[1] = 100;
//  sat_range_obj[0] = 135; sat_range_obj[1] = 255;

// red apple
//  hue_range_obj[0] = 106; hue_range_obj[1] = 140;
//  sat_range_obj[0] = 130; sat_range_obj[1] = 209;

// blue screwdriver
//  hue_range_obj[0] = 0; hue_range_obj[1] = 54;
//  sat_range_obj[0] = 140; sat_range_obj[1] = 184;

// orange
//  hue_range_obj[0] = 100; hue_range_obj[1] = 107;
//  sat_range_obj[0] = 163; sat_range_obj[1] = 255;

// yellow sponge
//  hue_range_obj[0] = 95; hue_range_obj[1] = 106;
//  sat_range_obj[0] = 110; sat_range_obj[1] = 186;

  cv::Mat img_rgb(480,640,CV_8UC3);
  cv::Mat seg_mask(480,640,CV_8UC1);
  cv::Rect box_obj;

  while(true)
  {
      sem_wait(&lock_t2);

      img_rgb = rgb_global.clone();
      segmentHSVEDIT(rgb_global, seg_mask,
    		         hue_range_obj[1], hue_range_obj[0],
    		         sat_range_obj[1], sat_range_obj[0]);
      noiseRemove(seg_mask,mask_obj_global,box_obj);
      object_blob_global = box_obj;

#ifdef FLAG_OBJECT
      cv::Mat rgb_tmp = cv::Mat::zeros(480,640, CV_8UC3);
      rgb_global.copyTo(rgb_tmp, mask_obj_global);
      cv::imshow("rgb_o",rgb_tmp); cvWaitKey(1);
#endif

      sem_post(&mutex_t5);
      sem_post(&lock_t1);
  }
  return 0;
}

//====================================================================================================================================
// [THREAD 3 : HAND DETECTOR]**************************************************
void* handDetector(void* arg)
{
  // Crop Threshold
  int hue_range_hand[2], sat_range_hand[2];
  hue_range_hand[0] = 102; hue_range_hand[1] = 122;
  sat_range_hand[0] = 69 ; sat_range_hand[1] = 150;

  // Variable init
  cv::Mat seg_mask(480,640,CV_8UC1);
  cv::Mat img_no_head = cv::Mat::zeros(480,640,CV_8UC3);
  cv::Rect box_hand;

  sleep(1);

  while(true)
  {
      sem_wait(&lock_t3);

      rgb_global.rowRange(225,480).copyTo(img_no_head.rowRange(225,480));
      segmentHSVEDIT(img_no_head, seg_mask,
    		         hue_range_hand[1], hue_range_hand[0],
    		         sat_range_hand[1], sat_range_hand[0]);
      noiseRemoveBox(seg_mask,mask_hand_global,box_hand);

#ifdef FLAG_HAND
      cv::Mat rgb_tmp = cv::Mat::zeros(480,640, CV_8UC3);
      rgb_global.copyTo(rgb_tmp, mask_hand_global);
      cv::imshow("rgb_h",rgb_tmp); cvWaitKey(1);
#endif

      sem_post(&mutex_t5);
      sem_post(&lock_t1);
  }
  return 0;
}

//====================================================================================================================================
// [THREAD 4 : FACE DETECTOR]**************************************************
void* faceDetector(void* arg)
{

  //Load the cascade for face detector
  std::string face_cascade_name = "lbpcascade_frontalface.xml";
  //std::string face_cascade_name = "haarcascade_frontalface_alt_tree.xml";
  cv::CascadeClassifier face_cascade;
  if( !face_cascade.load( face_cascade_name ) )
    { printf("--(!)Error loading face cascade\n");}

  cv::Mat kinect_rgb_img(480,640,CV_8UC3);
  cv::Mat kinect_rgb_img_face(480,640,CV_8UC3);

  std::vector<cv::Rect> faces;
  float frame_tmp = 0.0;

  sleep(1);

  while(true){   
    if(frame_number_global-10 > frame_tmp){
      frame_tmp = frame_number_global;
      kinect_rgb_img = rgb_global.clone();
      faces = detectFaceAndEyes(kinect_rgb_img,face_cascade);
      if(faces.size()==1)
        faces_global = cloud_global.at<cv::Vec3f>(faces[0].y+(faces[0].height/2),faces[0].x+(faces[0].width/2));
      //printf("     %d     ",faces.size());

#ifdef FLAG_FACE
      for( size_t i = 0; i < faces.size(); i++ ){
        cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        cv::ellipse( kinect_rgb_img, center, cv::Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, cv::Scalar( 255, 255, 255 ), -1, 1, 0 );          
      }
      cv::imshow("rgb_f",kinect_rgb_img); cvWaitKey(1);
#endif

    }
  }
  return 0;
}

//====================================================================================================================================
// [THREAD 5 : CONTACT DETECTOR]***********************************************
void* contactDetector(void* arg)
{
  cv::Mat img_depth(480,640,CV_8UC1);
  cv::Mat img_sub(480,640,CV_8UC1);
  cv::Mat img_diff(480,640,CV_8UC1);
  cv::Rect blob_def;
  cv::Mat cloud_mask,cloud_mask2;

  float contact_sub, contact_diff;
  int begin, ends, counter, contact_counter;
  bool flag = true;
  bool flag_contact_obj = false;

  Ipp8u ippi_depth_default[640*480] = {0};
  Ipp8u ippi_depth_tmp[640*480] = {0};
  Ipp8u ippi_depth_image[640*480] = {0};
  IppiSize roi_size;      roi_size.width = 640;    roi_size.height = 480;
  IppiSize roi_size_blob; roi_size_blob.width = 1; roi_size_blob.height = 1;
  IppiPoint roi_point;    roi_point.x = 0;         roi_point.x = 0;

  sleep(1);

  while(true)
  {
      sem_wait(&lock_t5);
      sem_wait(&mutex_t5);
      sem_wait(&mutex_t5);

     //[KINECT DEPTH]*********************************************************
      u_char      *ptr=img_depth.data;
      uint16_t    *depth=(uint16_t*)depth_global.data;
      for(int i=0;i<640*480;i++) ptr[i] = depth[i]/2048.00 * 255;
      //*********************************************************[KINECT DEPTH]

      //[DEFAULT SCENE]********************************************************
      if(flag){
        ippiCopy_8u_C1R(img_depth.data,640,
                        ippi_depth_default,640,roi_size);
        roi_size_blob.width  = object_blob_global.size().width ;
        roi_size_blob.height = object_blob_global.size().height;
        roi_point.x = object_blob_global.x;
        roi_point.y = object_blob_global.y;
        begin = 640*roi_point.y+roi_point.x;
        ends  = 640*(roi_point.y+roi_size_blob.height)+
                    (roi_point.x+roi_size_blob.width);
        flag = false;
      }
      //********************************************************[DEFAULT SCENE]

      //[OBJECT POINT]*********************************************************
      cloud_global.copyTo(cloud_mask,mask_obj_global); //taking the obj only
      cloud_mask(object_blob_global).copyTo(cloud_mask2); // reducing the search area
      //cloud_global(object_blob_global).copyTo(cloud_mask);
      pointCloudTrajectory(cloud_mask2,single_point_obj_global,eig_val_global);
      cloud_mask.release();
      cloud_mask2.release();
      //*********************************************************[OBJECT POINT]

      //[OBJECT CONTACT]*******************************************************    
      if(contactCheck(mask_hand_global,object_blob_global)){
        if(!contact_obj){
          ippiSub_8u_C1RSfs( img_depth.data+640*roi_point.y+roi_point.x,640,
                             ippi_depth_default+640*roi_point.y+roi_point.x,640,
                             ippi_depth_tmp,640,
                             roi_size_blob,0);
          ippiMulC_8u_C1RSfs(ippi_depth_tmp,640,
                             1,
                             ippi_depth_image+640*roi_point.y+roi_point.x,640,
                             roi_size,0);
          ippiCopy_8u_C1R(ippi_depth_image,640,img_sub.data,640,roi_size);

          contact_sub  = 0.0; counter = 0;
          for(int i=begin;i<ends;i++){
            if(img_sub.data[i] > 0){
              counter += 1;
              contact_sub += img_sub.data[i];
            }
          } 
          contact_sub = contact_sub/counter;

          if(contact_sub>0 && contact_sub< 15){ 
            contact_counter += 1;
          }
          else contact_counter = 0;
        }
        else contact_counter = 3;     
      }
      else contact_counter = 0;

      if(contact_counter > 2) contact_obj = true; else contact_obj = false;

      if(object_blob_global.y < 241) {contact_obj = true;} // face prevention
      //*******************************************************[OBJECT CONTACT]

      //[MARKER CONTACT]******************************************************* 
      if(flag_marker){
        contact_marker = markerContact(marker_center_global, 
                                       plane_global, 
                                       single_point_obj_global);
      }
      //*******************************************************[MARKER CONTACT]

      printf("CONTACT : %d     %d", contact_obj, contact_marker);
//      printf("CONTACT : %d     %d       CONTACTVAL : %f", contact_obj, contact_marker, contact_sub);

      sem_post(&mutex_t6);
      sem_post(&lock_t1);
  }
  return 0;
}

//====================================================================================================================================
// [THREAD 6 : WRITE DATA]*****************************************************
void* writeData(void* arg)
{
  cv::Mat img_tmp(480,640,CV_8UC1);
  cv::Vec3f single_point_obj;
  std::remove("traj_data.txt");
  std::ofstream write_file;
  cv::Mat cloud_mask,cloud_mask2;

  float window1 = 5.0;
  float window2 = 3.0;
  std::vector<cv::Vec3f> points(window1);
  std::vector<cv::Vec3f> pca_points(7);
  std::vector<cv::Vec3f> points_sinuosity(7);
  std::vector<cv::Vec3f> velocities(window2);
  std::vector<float> speeds(window2);
  std::vector<float> slides(window1);
  std::vector<double> eig_vals1(window1);
  std::vector<double> eig_vals2(window1);
  cv::Vec3f p1(0,0,0), p2(0,0,0);
  cv::Vec3f vel1(0,0,0), vel2(0,0,0);
  float spd1 = 0.0, spd2 = 0.0;
  float sinuosity1 = 0.0;
  float sinuosity2 = 0.0;
  float sinuosity3 = 0.0;
  double EV1, EV2;
  bool tilt = false;

  float slide = 0.0;

  cv::Vec3f surface(0,0,0);

  int pca_window = 9;
  cv::Mat pca_data_pts = cv::Mat(pca_window, 3, CV_64FC1);
  double constraints = 0.0;

  int loc_known, loc_pred;
  int loc_num = 3;
  std::vector<cv::Vec3f> locations(loc_num);
  std::vector<double> locations_limit(loc_num);
  std::vector<double> dist_p_tmp(loc_num);
  std::vector<double> dist_p_tmp2(loc_num);
  std::vector<double> dist_p_loc(loc_num);
  std::vector<double> angle_p_loc(loc_num);
  std::vector<double> pred_p_loc(loc_num);
  double pred_p_loc_tmp = 0.0;
  locations_limit[0] = 0.15;
  locations_limit[1] = 0.15;
  locations_limit[2] = 0.15;

  int action = 0;
  std::vector<std::string> action_name(10);
  action_name[0] = "NULL    ";
  action_name[1] = "MOVING  ";
  action_name[2] = "DRINKING";
  action_name[3] = "DISPOSE "; 
  action_name[4] = "FILLING "; 
  action_name[5] = "TILTING "; 
  action_name[6] = "SLIDING "; 
  action_name[7] = " "; 
  action_name[8] = " "; 
  action_name[9] = " "; 

  sleep(1);

  while(true)
  {
    sem_wait(&lock_t6);
    sem_wait(&mutex_t6);
    
    for(int i=0;i<window1-1;i++) points[i] = points[i+1];
    points[window1-1] = single_point_obj_global;
    p1 = p2; // last point
    p2 = movingAveragePoint(points,window1);

    for(int i=0;i<window2-1;i++) velocities[i] = velocities[i+1];
    velocities[window2-1] = pointToVelocity(p1,p2);
    vel1 = vel2; // last point
    vel2 = movingAveragePoint(velocities,window2);

    for(int i=0;i<window2-1;i++) speeds[i] = speeds[i+1];
    speeds[window2-1] = pointToSpeed(p1,p2);
    spd1 = spd2;
    spd2 = movingAverageFloat(speeds,window2);

    for(int i=0;i<window1-1;i++) eig_vals1[i] = eig_vals1[i+1];
    eig_vals1[window1-1] = eig_val_global[0];
    EV1 = movingAverageDouble(eig_vals1,window1);
    for(int i=0;i<window1-1;i++) eig_vals2[i] = eig_vals2[i+1];
    eig_vals2[window1-1] = eig_val_global[1];
    EV2 = movingAverageDouble(eig_vals2,window1);

    for(int i=0;i<7-1;i++) points_sinuosity[i] = points_sinuosity[i+1];
    points_sinuosity[7-1] = p2;
    sinuosity1 = (norm(points_sinuosity[4]-points_sinuosity[5])  + 
                  norm(points_sinuosity[5]-points_sinuosity[6])) / 
                 (norm(points_sinuosity[4]-points_sinuosity[6]));
    sinuosity2 = (norm(points_sinuosity[2]-points_sinuosity[4])  + 
                  norm(points_sinuosity[4]-points_sinuosity[6])) / 
                 (norm(points_sinuosity[2]-points_sinuosity[6]));
    sinuosity3 = (norm(points_sinuosity[0]-points_sinuosity[3])  + 
                  norm(points_sinuosity[3]-points_sinuosity[6])) / 
                 (norm(points_sinuosity[0]-points_sinuosity[6]));

    locations[0] = faces_global;
    locations[1] = marker_center_global[0];
    locations[2] = marker_center_global[1];

    loc_known = 0;
    loc_pred = 0;

    for(int i=0;i<loc_num;i++){      
      dist_p_tmp[i] = norm(locations[i]-p2); // testing
      dist_p_tmp2[i] = (norm(locations[i]-p2)-locations_limit[i]); // testing
      angle_p_loc[i] = std::atan2(norm(crossProd(vel2,(locations[i]-p2))),
                                  dotProd(vel2,(locations[i]-p2)));
      angle_p_loc[i] = 1 - (angle_p_loc[i]/M_PI);
      dist_p_loc[i] = exp(-5*(norm(locations[i]-p2)-locations_limit[i])*(norm(locations[i]-p2)-locations_limit[i]));
    }


    surface[0] = plane_global[0];
    surface[1] = plane_global[1];
    surface[2] = plane_global[2];
    slide = norm(crossProd(vel2,surface))/(norm(vel2)*norm(surface));    
    for(int i=0;i<window1-1;i++) slides[i] = slides[i+1];
    slides[window1-1] = slide;
    slide = movingAverageFloat(slides,window1);
    if(p2[0]*surface[0] + 
       p2[1]*surface[1] + 
       p2[2]*surface[2] - plane_global[3] > 0.1)
      slide = 0.0;





    for(int i=0;i<pca_window-1;i++) pca_points[i] = pca_points[i+1];
    pca_points[pca_window-1] = p2;

    for(int i=0;i<pca_window-1;i++){
      pca_points[i] = pca_points[i+1];
      pca_data_pts.at<double>(i, 0) = pca_points[i][0];
      pca_data_pts.at<double>(i, 1) = pca_points[i][1];
      pca_data_pts.at<double>(i, 2) = pca_points[i][2];
    }
    pca_points[pca_window-1] = p2;
    pca_data_pts.at<double>(pca_window-1, 0) = pca_points[pca_window-1][0];
    pca_data_pts.at<double>(pca_window-1, 1) = pca_points[pca_window-1][1];
    pca_data_pts.at<double>(pca_window-1, 2) = pca_points[pca_window-1][2];
    //Perform PCA analysis
    cv::PCA pca_analysis(pca_data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);
    //Store the eigenvalues and eigenvectors
    std::vector<cv::Point3d> eigen_vecs(3);
    std::vector<double> eigen_val(3);
    for (int i=0;i<3;++i){
      eigen_vecs[i] = cv::Point3d(pca_analysis.eigenvectors.at<double>(i, 0),
                                  pca_analysis.eigenvectors.at<double>(i, 1),
                                  pca_analysis.eigenvectors.at<double>(i, 2));
      eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
    }
    if(eigen_val[0]>0 && eigen_val[0]>0)
      constraints = (eigen_val[0] / eigen_val[2]);


//=========================================================================================================================================
    pred_p_loc_tmp = 0.0;
    // moving
    if(spd2 > 0.003){ // 0.003 is hard coded, need more evaluation ####
      action = 1;
      
      //[SLIDING PREDICTION]
      if(slide > 0.9) action = 6;

      //[TILTING PREDICTION]
      //if(EV1 > 0.00061 && EV2 < 0.00033) tilt = true;
//      if(EV1/EV2 > 1.7) tilt = true;
//      else tilt = false;
//      if(tilt) action = 5;

      //[LOCATION PREDICTION]
      for(int i=0;i<loc_num;i++){
        pred_p_loc[i] = (0.5 * angle_p_loc[i]) + (0.5 * dist_p_loc[i]);
        if (pred_p_loc[i] > pred_p_loc_tmp){
          loc_pred = i;
          pred_p_loc_tmp = pred_p_loc[i];
        }
        if(dist_p_tmp[i] < locations_limit[i] && spd2 < 0.02)
          action = 2 + i;
      }
    }
    // stationary
    else{ 
      action = 0;
    
      //[TILTING PREDICTION]
//      if(EV1/EV2 > 1.7) tilt = true;
//      else tilt = false;
//      if(tilt) action = 5;

      //[LOCATION PREDICTION]
      for(int i=0;i<loc_num;i++){  
        pred_p_loc[i] = (0.0 * angle_p_loc[i]) + (1.0 * dist_p_loc[i]);
        if (pred_p_loc[i] > pred_p_loc_tmp){
          loc_pred = i;
          pred_p_loc_tmp = pred_p_loc[i];
        }
        if(dist_p_tmp[i] < locations_limit[i])
          action = 2 + i;
      }
    }




#ifdef FLAG_WRITE
    if(flag_marker){
    // write values into data.txt
    std::ofstream write_file("traj_data.txt", std::ios::app);
    write_file << frame_number_global << ","
               << plane_global[0] << ","
               << plane_global[1] << ","
               << plane_global[2] << ","
               << plane_global[3] << ","
               << marker_center_global[0][0] << ","
               << marker_center_global[0][1] << ","
               << marker_center_global[0][2] << ","
               << marker_center_global[1][0] << ","
               << marker_center_global[1][1] << ","
               << marker_center_global[1][2] << ","
               << contact_marker << ","
               << contact_obj << ","
               << single_point_obj_global[0] << ","
               << single_point_obj_global[1] << ","
               << single_point_obj_global[2] << ","
               << faces_global[0] << ","
               << faces_global[1] << ","
               << faces_global[2] << ","
               << dist_p_loc[0] << ","
               << dist_p_loc[1] << ","
               << dist_p_loc[2] << ","
               << angle_p_loc[0] << ","
               << angle_p_loc[1] << ","
               << angle_p_loc[2] << ","
               << pred_p_loc[0] << ","
               << pred_p_loc[1] << ","
               << pred_p_loc[2] << ","
               << spd2 << ","
               << sinuosity1 << ","
               << sinuosity2 << ","
               << sinuosity3 << ","
               << eig_val_global[0] << ","
               << eig_val_global[1] << ","
               << eig_val_global[2] 
               << "\n";
    }
#endif


//    std::cout << "   " << dist_p_loc[0] << " , " << dist_p_loc[1] << " , " << dist_p_loc[2] ;
//    printf("  %f  %f  %f  ",pred_p_loc[0],pred_p_loc[1],pred_p_loc[2]);
    printf("  %s  %s\n",action_name[action].c_str(),action_name[loc_pred+2].c_str());

    sem_post(&lock_t1);
  }
  return 0;
}

//====================================================================================================================================

int main()
{
	
  // The kinect3d object        
  cv::VideoCapture kinect(CV_CAP_OPENNI2); printf("Starting Kinect ...\n");

  // Depth value processing
  uint16_t mGamma[2048];
  for( int i=0;i<2048;++i )
    { float v=i/2048.0; v=powf(v, 3)*6; mGamma[i]=v*6*256;}   

  // Run the visualization
#ifdef FLAG_DEPTH
  cv::namedWindow("depth");
#endif
#ifdef FLAG_RGB
  cv::namedWindow("rgb");
#endif
#ifdef FLAG_HAND
  cv::namedWindow("rgb_h");
  cvMoveWindow("rgb_h",0,0);
#endif
#ifdef FLAG_OBJECT
  cv::namedWindow("rgb_o");
  cvMoveWindow("rgb_o",0,490);  
#endif
#ifdef FLAG_FACE
  cv::namedWindow("rgb_f");
  cvMoveWindow("rgb_f",650,0);  
#endif
#ifdef FLAG_PLANE
  cv::namedWindow("plane");
#endif
#ifdef FLAG_MARKER
  cv::namedWindow("rgb_m");
#endif

  cv::Mat rgb_image,disp_depth,point_cloud,depth_image(480,640,CV_8UC3);

#ifdef FLAG_THREAD
  // Start multithread
  pthread_t thread_kinectGrab,
            thread_objDetector,
            thread_handDetector,
            thread_faceDetector,
            thread_contactDetector,
            thread_writeData;

  sem_init(&lock_t1, 0, MAX);
  sem_init(&lock_t2, 0, 0);
  sem_init(&lock_t3, 0, 0);
//  sem_init(&lock_t4, 0, 0);
  sem_init(&lock_t5, 0, 0);
  sem_init(&lock_t6, 0, 0);
  sem_init(&mutex_t1, 0, 1);
  sem_init(&mutex_t5, 0, 0);
  sem_init(&mutex_t6, 0, 0);

  pthread_attr_t attr;
  cpu_set_t cpus;
  pthread_attr_init(&attr);

  CPU_ZERO(&cpus);
  CPU_SET(1, &cpus);
  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
  pthread_create(&thread_kinectGrab, &attr, kinectGrab, &kinect);

  CPU_ZERO(&cpus);
  CPU_SET(2, &cpus);
  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
  pthread_create(&thread_objDetector, &attr, objectDetector, NULL);

  CPU_ZERO(&cpus);
  CPU_SET(3, &cpus);
  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);    
  pthread_create(&thread_handDetector, &attr, handDetector, NULL);

  CPU_ZERO(&cpus);
  CPU_SET(4, &cpus);
  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);    
  pthread_create(&thread_faceDetector, &attr, faceDetector, NULL);

  CPU_ZERO(&cpus);
  CPU_SET(5, &cpus);
  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);    
  pthread_create(&thread_contactDetector, &attr, contactDetector, NULL);

  CPU_ZERO(&cpus);
  CPU_SET(6, &cpus);
  pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);    
  pthread_create(&thread_writeData, &attr, writeData, NULL);

  pthread_join(thread_kinectGrab, NULL);
  pthread_join(thread_objDetector, NULL);
  pthread_join(thread_handDetector, NULL);
//  pthread_join(thread_faceDetector, NULL);
  pthread_join(thread_contactDetector, NULL);
  pthread_join(thread_writeData, NULL);

  //printf("MAIN THREAD ON CORE : %d\n",sched_getcpu()); 

#else 
  while(true)
  {
    kinect.grab();
    kinect.retrieve(rgb_global,CV_CAP_OPENNI_BGR_IMAGE);
//    kinect.retrieve(depth_global,CV_CAP_OPENNI_DEPTH_MAP);
//    kinect.retrieve(cloud_global,CV_CAP_OPENNI_POINT_CLOUD_MAP);

    cv::imshow("rgb_global",rgb_global); cvWaitKey(1);

    if(flag_thres)
    {
      cv::imwrite( "test.png" , rgb_global );
      int hue_range[2], sat_range[2];
      cv::Mat img = cv::imread("test.png");
      getColorThreshold(img, hue_range, sat_range);
      printf("Final Calibration values:\nhue = %d %d\nsat = %d %d\n",
              hue_range[0],hue_range[1],
              sat_range[0],sat_range[1]);
      cv::imshow("rgb",img);
      std::cout << "press <s> to stop kinect.\n";
      char k = cv::waitKey(0); if (k == 's') break; 
    }   
  }
#endif 

  return 0;
}























//====================================================================================================================================
/*
void* faceDetector(void* arg)
{

  //Load the cascade for face detector
  std::string face_cascade_name = "lbpcascade_frontalface.xml";
  std::string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
  cv::CascadeClassifier face_cascade;
  cv::CascadeClassifier eyes_cascade;
  if( !face_cascade.load( face_cascade_name ) )
    { printf("--(!)Error loading face cascade\n");}
  if( !eyes_cascade.load( eyes_cascade_name ) )
    { printf("--(!)Error loading eyes cascade\n");}

  cv::Mat kinect_rgb_img(480,640,CV_8UC3);

  sleep(1);

  while(true)
  {
	rgb_global.clone().copyTo(kinect_rgb_img);
    detectAndDisplay( kinect_rgb_img , mask_head_global, face_cascade , eyes_cascade);
    kinect_rgb_img.clone().copyTo(img_no_head);
  }
  
  return 0;
}
*/
//====================================================================================================================================


/*
#include <vtkPointSource.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
 
int main(int, char *[])
{

  cv::VideoCapture * kinect = reinterpret_cast<cv::VideoCapture *>(v_kinect);

  cv::Mat cloud_global;

    kinect.grab();
    kinect.retrieve(cloud_global,CV_CAP_OPENNI_POINT_CLOUD_MAP);



  // Create a point cloud
  vtkSmartPointer<vtkPointSource> pointSource =
    vtkSmartPointer<vtkPointSource>::New();
  pointSource->SetCenter(0.0, 0.0, 0.0);
  pointSource->SetNumberOfPoints(50);
  pointSource->SetRadius(5.0);
  pointSource->Update();
 
  // Create a mapper and actor
  vtkSmartPointer<vtkPolyDataMapper> mapper =
    vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(pointSource->GetOutputPort());
 
  vtkSmartPointer<vtkActor> actor =
    vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);
 
  // Create a renderer, render window, and interactor
  vtkSmartPointer<vtkRenderer> renderer =
    vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow =
    vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
    vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
 
  // Add the actor to the scene
  renderer->AddActor(actor);
  renderer->SetBackground(.3, .6, .3); // Background color green
 
  // Render and interact
  renderWindow->Render();
  renderWindowInteractor->Start();
 
  return EXIT_SUCCESS;
}
*/

