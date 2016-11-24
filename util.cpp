#include "util.h"


/******************************************************************************
Classes
******************************************************************************/
//[ATT] read_img and read_label are not optimized
Dataset::Dataset(	std::string dir,
					bool read_label,
					bool read_img,
					std::string img_extension){

	char *dir_char = new char[dir.length()+1];
	strcpy(dir_char, dir.c_str());
	_dir = dir;

	DIR* dir_ = opendir(dir_char);
	struct dirent* read_dir;

	while((read_dir = readdir(dir_)) != NULL){

		// disregard "." and ".."
		if(	!strcmp(read_dir->d_name, ".") ||!strcmp(read_dir->d_name, ".."))
			continue;

		std::string fn = std::string(read_dir->d_name);
		std::size_t found_extension = fn.find(img_extension);

		// disregard non-img_extension files
		if(found_extension==std::string::npos)
			continue;

		//
		char* name = new char[256];
		sprintf(name, "%s/%s",dir_char ,read_dir->d_name);

		if(read_img)
			_img.push_back( cv::imread(name));

		_fn.push_back(fn);

		if(read_label)
			_label.push_back(std::atoi(fn.substr(0,3).c_str()));


		delete name;

	}

	_size=_fn.size();
}
/******************************************************************************
Functions
******************************************************************************/

//[TOOLS]**********************************************************************

cv::Vec3f crossProd(cv::Vec3f A, cv::Vec3f B){
  cv::Vec3f C;
  C[0] = A[1]*B[2] - A[2]*B[1]; 
  C[1] = A[2]*B[0] - A[0]*B[2]; 
  C[2] = A[0]*B[1] - A[1]*B[0];
  if(C[0]*C[0]+C[1]*C[1]+C[2]*C[2] == 0){ // prevent degenerate case
    printf("WARNING : VECTORS ARE COLLINEAR !!!\n");
    C[0]=0; C[1]=0; C[2]=0;
  }
  if(A[0] == 0 && A[1] == 0 && A[2] == 0) 
    printf("WARNING : VECTOR A IS A ZERO VECTOR !!!\n");
  if(B[0] == 0 && B[1] == 0 && B[2] == 0) 
    printf("WARNING : VECTOR B IS A ZERO VECTOR !!!\n");
  return C;
}

float dotProd(cv::Vec3f A, cv::Vec3f B){
  float ans;
  cv::Vec3f C;
  C[0] = A[0]*B[0]; 
  C[1] = A[1]*B[1]; 
  C[2] = A[2]*B[2];
  ans = C[0]+C[1]+C[2];
  if(A[0] == 0 && A[1] == 0 && A[2] == 0) 
    printf("WARNING : VECTOR A IS A ZERO VECTOR !!!\n");
  if(B[0] == 0 && B[1] == 0 && B[2] == 0) 
    printf("WARNING : VECTOR B IS A ZERO VECTOR !!!\n");
  return ans;
}

cv::Vec3f normalization3D(cv::Vec3f vec){
  float length = sqrt(vec[0]*vec[0]+
                      vec[1]*vec[1]+
                      vec[2]*vec[2]);
  cv::Vec3f vec_normed(vec[0]/length,vec[1]/length,vec[2]/length);
  return vec_normed;
}

cv::Vec3f computePlane(cv::Vec3f A, cv::Vec3f B, cv::Vec3f C){
  cv::Vec3f N_norm;
  cv::Vec3f N = crossProd((A - B),(B - C)); //perform cross product of two lines on plane 
  N_norm = normalization3D(N);
  return N_norm;
}

void normalPlaneCheck(cv::Vec4f &plane_equation){
  cv::Vec3f p1(0,0,1); 
  if (plane_equation[0]*p1[0]+
      plane_equation[1]*p1[1]+
      plane_equation[2]*p1[2]-
      plane_equation[3]< 0)
  {
    plane_equation = (-1) * plane_equation;
  }
}

void tic(){
    tictoc_stack.push(clock());
}

void toc(){
    std::cout << "Time elapsed: "
              << ((double)(clock() - tictoc_stack.top())) / CLOCKS_PER_SEC
              << std::endl;
    tictoc_stack.pop();
}

cv::Mat imgResize(cv::Mat src, int width, int height){
  cv::Mat dst;
  cv::Size size;
  size.width = width;
  size.height = height;
  cv::resize(src, dst, size, CV_INTER_NN);
  return dst;
}

bool fexists(const char *filename){
  std::ifstream ifile(filename);
  return ifile;
}

bool fexists(const std::string& filename){
  std::ifstream ifile(filename.c_str());
  return ifile;
}

bool copyFile(const char *SRC, const char* DEST){
  std::ifstream src(SRC, std::ios::binary);
  std::ofstream dest(DEST, std::ios::binary);
  dest << src.rdbuf();
  return src && dest;
}

void mouseCallBackHist(	int eventcode,int x,int y,int flags,void* data){
  if(eventcode == CV_EVENT_LBUTTONDOWN){
    MouseCallBackHistData* mouse_data = (MouseCallBackHistData*)data;
    int x_hist = floor((float)x / (*mouse_data)._hscale);
    int y_hist = floor((float)y / (*mouse_data)._sscale);
    if (x_hist >= (*mouse_data)._hist.cols)
      x_hist = (*mouse_data)._hist.cols - 1;
    if (y_hist >= (*mouse_data)._hist.rows)
      y_hist = (*mouse_data)._hist.rows - 1;
    int h = round(x_hist* (*mouse_data)._hbin_scale);
    int s = round(y_hist* (*mouse_data)._sbin_scale);
    float i = (*mouse_data)._hist.at<float>(x_hist, y_hist);
    mouse_data->_h_mean = h;
    mouse_data->_s_mean = s;
    std::cout << "H: " << h 
              << " +/- " << (*mouse_data)._hbin_scale
              << ", S: " << s 
              << " +/- " << (*mouse_data)._sbin_scale
              << ", Intensity: " << i << std::endl;
  }
}

void trackbarCallBackCalib_H_top(int trackpos, void* data){
  TrackbarCBDataCalib* tb = (TrackbarCBDataCalib*)data;
  tb->_H_top = round(trackpos*179.0 / 100);
  segmentHSV( tb->_hsv, //in
              tb->_seg_in_rgb, //in
              tb->_seg_mask, //out
              tb->_seg_out_rgb, //out
              tb->_H_top, //in
              tb->_H_bot, //in
              tb->_S_top, //in
              tb->_S_bot); //in
  cv::imshow(tb->_wn_seg_mask, tb->_seg_mask);
  cv::imshow(tb->_wn_seg_rgb, tb->_seg_out_rgb);
  std::cout << "[" << "H_top=" << tb->_H_top << ", "
                   << "H_bot=" << tb->_H_bot << ", "
                   << "S_top=" << tb->_S_top << ", "
                   << "S_bot=" << tb->_S_bot << "]\n";
}

void trackbarCallBackCalib_H_bot(int trackpos, void* data){
  TrackbarCBDataCalib* tb = (TrackbarCBDataCalib*)data;
  tb->_H_bot = round(trackpos*179.0 / 100);
  segmentHSV( tb->_hsv, //in
              tb->_seg_in_rgb, //in
              tb->_seg_mask, //out
              tb->_seg_out_rgb, //out
              tb->_H_top, //in
              tb->_H_bot, //in
              tb->_S_top, //in
              tb->_S_bot); //in
  cv::imshow(tb->_wn_seg_mask, tb->_seg_mask);
  cv::imshow(tb->_wn_seg_rgb, tb->_seg_out_rgb);
  std::cout << "[" << "H_top=" << tb->_H_top << ", "
                   << "H_bot=" << tb->_H_bot << ", "
                   << "S_top=" << tb->_S_top << ", "
                   << "S_bot=" << tb->_S_bot << "]\n";
}

void trackbarCallBackCalib_S_top(int trackpos, void* data){
  TrackbarCBDataCalib* tb = (TrackbarCBDataCalib*)data;
  tb->_S_top = round(trackpos*255.0 / 100);
  segmentHSV( tb->_hsv, //in
              tb->_seg_in_rgb, //in
              tb->_seg_mask, //out
              tb->_seg_out_rgb, //out
              tb->_H_top, //in
              tb->_H_bot, //in
              tb->_S_top, //in
              tb->_S_bot); //in
  cv::imshow(tb->_wn_seg_mask, tb->_seg_mask);
  cv::imshow(tb->_wn_seg_rgb, tb->_seg_out_rgb);
  std::cout << "[" << "H_top=" << tb->_H_top << ", "
                   << "H_bot=" << tb->_H_bot << ", "
                   << "S_top=" << tb->_S_top << ", "
                   << "S_bot=" << tb->_S_bot << "]\n";
}

void trackbarCallBackCalib_S_bot(int trackpos, void* data){
  TrackbarCBDataCalib* tb = (TrackbarCBDataCalib*)data;
  tb->_S_bot = round(trackpos*255.0 / 100);
  segmentHSV( tb->_hsv, //in
              tb->_seg_in_rgb, //in
              tb->_seg_mask, //out
              tb->_seg_out_rgb, //out
              tb->_H_top, //in
              tb->_H_bot, //in
              tb->_S_top, //in
              tb->_S_bot); //in
  cv::imshow(tb->_wn_seg_mask, tb->_seg_mask);
  cv::imshow(tb->_wn_seg_rgb, tb->_seg_out_rgb);
  std::cout << "[" << "H_top=" << tb->_H_top << ", "
                   << "H_bot=" << tb->_H_bot << ", "
                   << "S_top=" << tb->_S_top << ", "
                   << "S_bot=" << tb->_S_bot << "]\n";
}

//**********************************************************************[TOOLS]


//====================================================================================================================================

std::vector<aruco::Marker> arucoMarkerDetector
  (cv::Mat &rgb, bool write_id, bool display_id){

  aruco::MarkerDetector MDetector;
  std::vector<aruco::Marker> Markers;
  MDetector.detect(rgb,Markers);  
  for(int i=0;i<Markers.size();i++){
    aruco::Marker marker_tmp = Markers[i];    
    Markers[i].draw(rgb,cv::Scalar(255,255,0),2,display_id); 
    //circle(rgb, Markers[i].getCenter(), 3, cv::Scalar(255,0,255), -1);
    if(write_id)
      printf("MARKER ID : %d\n",Markers[i].id);
  }
  return Markers;
}



//====================================================================================================================================

void getColorThreshold(cv::Mat src, int (&hue_range)[2], int (&sat_range)[2]){
  // check and warn if type of src is not CV_8UC3
  if (src.type() != CV_8UC3){
    std::cout
      << "Error: [getColorThreshold()]: src must have type CV_8UC3!\n";
    std::cout << "press any key to exit\n";
    cv::waitKey(0);
    exit(1);
  }
  cv::imshow("src in RGB", src);
  // preproc do medianfilter*************************************************
  cv::Mat src_median_blurred; // median burred src in rgb
  cv::medianBlur(src, src_median_blurred, 5);
  cv::imshow("Blurred", src_median_blurred); //just for debugging

  // RGB -> HSV H=[0...179] S,V=[0...255]
  cv::Mat src_hsv; // median blurred src hsv
  cv::cvtColor(src_median_blurred, src_hsv, CV_RGB2HSV);

  // compute histogram*******************************************************
  int hbins = 30; // please ensure that this is a modulo of the whole range
  int sbins = 16; // please ensure that this is a modulo of the whole range
  int histSize[] = { hbins, sbins };
  float hranges[] = { 0, 180 };
  float sranges[] = { 0, 256 };
  const float* ranges[] = { hranges, sranges };
  int channels[] = { 0, 1 };
  cv::MatND hist;
  calcHist(&src_hsv, 1, channels, cv::Mat(), // do not use mask
               hist, 2, histSize, ranges,
               true, // the histogram is uniform
               false);
  double maxVal = 0;
  minMaxLoc(hist, 0, &maxVal, 0, 0);
  int scale = 20;
  std::vector<cv::Mat> col_hist(3);
  col_hist[0] = cv::Mat::zeros(sbins*scale, hbins * scale, CV_8UC1); //H
  col_hist[1] = cv::Mat::zeros(sbins*scale, hbins * scale, CV_8UC1); //S
  col_hist[2] = cv::Mat::zeros(sbins*scale, hbins * scale, CV_8UC1); //V
  for (int h = 0; h < hbins; h++){
    for (int s = 0; s < sbins; s++)
    {
      float binVal = hist.at<float>(h, s);
      int v_intensity;
      // this is a hack to intensify smaller peaks because if the
      // background is big and uniform then there is a top heavy effect
      if ((((float)binVal) / ((float)maxVal)) > 0.05)
        v_intensity = 255;
      else
        v_intensity = cvRound(binVal * 255 / maxVal);
        cv::rectangle(col_hist[0], cv::Point(h*scale, s*scale),
              cv::Point((h + 1)*scale - 1, (s + 1)*scale - 1),
              cv::Scalar::all(h*(180 / hbins)),
              CV_FILLED);
        cv::rectangle(col_hist[1], cv::Point(h*scale, s*scale),
              cv::Point((h + 1)*scale - 1, (s + 1)*scale - 1),
              //cv::Scalar::all(s*(256/ sbins)),
              cv::Scalar::all(256),
              CV_FILLED);
        cv::rectangle(col_hist[2], cv::Point(h*scale, s*scale),
              cv::Point((h + 1)*scale - 1, (s + 1)*scale - 1),
              cv::Scalar::all(v_intensity),
              CV_FILLED);
    }
  }

  cv::Mat colored_hist_hsv;
  cv::merge(col_hist, colored_hist_hsv);
  cv::Mat colored_hist_rgb;
  cv::cvtColor(colored_hist_hsv, colored_hist_rgb, CV_HSV2RGB);
  std::string hist_win_name = "Colored H-S Histogram";
  cv::imshow(hist_win_name, colored_hist_rgb);
  double s = cv::sum(hist)[0];
  MouseCallBackHistData mouse_data;
  mouse_data._hscale = scale;
  mouse_data._sscale = scale;
  mouse_data._hbin_scale = (180 / hbins);
  mouse_data._sbin_scale = (256 / sbins);
  mouse_data._hist = hist/s;
  cv::setMouseCallback(hist_win_name, mouseCallBackHist, &mouse_data);
  std::cout << "Click on " << hist_win_name
            << "to read off all the hsv values from the console\n";
  std::cout << "press any key to go further\n";
  cv::waitKey(0);
  int mean_h = mouse_data._h_mean;
  int mean_s = mouse_data._s_mean;
  std::cout << "mean hue = " << mean_h 
            << ", mean saturation = " << mean_s << "\n\n";
  cv::destroyAllWindows();
  // calibration*************************************************************
  std::cout << "\n\n...calibration phase...\n\n";
  std::cout << "WARNING [calibration phase]: hue range is polar(cyclic)!!\n";
  cv::imshow("Median Blurred SRC", src_median_blurred); //just for debugging
  //int range_h[2] = { 0, 0 };
  //int range_s[2] = { 0, 0 };
  // create trackbar
  int H_top = round(mean_h * 100 / 180) + 5; // starting value for bar1
  int H_bot = round(mean_h * 100 / 180) - 5; // starting value for bar2
  int S_top = round(mean_s * 100 / 180) + 5; // starting value for bar3
  int S_bot = round(mean_s * 100 / 180) - 5; // starting value for bar4
  std::string calib_win_name = "Color Range Calibration";
  cv::namedWindow(calib_win_name, 1);
  TrackbarCBDataCalib tb_calib;
  tb_calib._hsv = src_hsv;
  tb_calib._seg_in_rgb = src_median_blurred;
  tb_calib._seg_mask = cv::Mat();
  tb_calib._seg_out_rgb = cv::Mat();
  tb_calib._H_top = mean_h + 5;
  tb_calib._H_bot = mean_h - 5;
  tb_calib._S_top = mean_s + 5;
  tb_calib._S_bot = mean_s - 5;
  tb_calib._wn_seg_mask = "seg_mask";
  tb_calib._wn_seg_rgb = "seg_rgb";
  cv::createTrackbar("H_top", calib_win_name, &H_top, 100,
    trackbarCallBackCalib_H_top, &tb_calib);
  cv::createTrackbar("H_bot", calib_win_name, &H_bot, 100,
    trackbarCallBackCalib_H_bot, &tb_calib);
  cv::createTrackbar("S_top", calib_win_name, &S_top, 100,
    trackbarCallBackCalib_S_top, &tb_calib);
  cv::createTrackbar("S_bot", calib_win_name, &S_bot, 100,
    trackbarCallBackCalib_S_bot, &tb_calib);
  cv::resizeWindow(calib_win_name, 500, 100);
  cv::waitKey(30);
  char stopkey = 'a';
  std::cout << "press <s> to stop calibrating.\n";
  segmentHSV(tb_calib._hsv, //in
             tb_calib._seg_in_rgb, //in
             tb_calib._seg_mask, //out
             tb_calib._seg_out_rgb, //out
             tb_calib._H_top, //in
             tb_calib._H_bot, //in
             tb_calib._S_top, //in
             tb_calib._S_bot); //in
  cv::imshow(tb_calib._wn_seg_mask, tb_calib._seg_mask);
  cv::imshow(tb_calib._wn_seg_rgb, tb_calib._seg_out_rgb);
  std::cout << "[" << "H_top=" << tb_calib._H_top << ", "
                   << "H_bot=" << tb_calib._H_bot << ", "
                   << "S_top=" << tb_calib._S_top << ", "
                   << "S_bot=" << tb_calib._S_bot << "]\n";
  while (stopkey != 's'){stopkey = cv::waitKey(0);}
  // extract calibrated values
  cv::destroyAllWindows();
  hue_range[0] = tb_calib._H_bot;
  hue_range[1] = tb_calib._H_top;
  sat_range[0] = tb_calib._S_bot;
  sat_range[1] = tb_calib._S_top;
}

//====================================================================================================================================

void segmentHSV(cv::Mat src_hsv, cv::Mat src_rgb,
                cv::Mat& seg_mask, cv::Mat& seg_rgb,
                int h_top, int h_bot, int s_top, int s_bot){
  std::vector<cv::Mat> splitted_HSV;
  cv::split(src_hsv, splitted_HSV);
  cv::Mat seg_mask1 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  cv::Mat seg_mask2 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  cv::Mat seg_mask3 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  cv::Mat seg_mask4 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  // Thresholding
  seg_mask1 = splitted_HSV[0] <= (h_top);
  seg_mask2 = splitted_HSV[0] >= (h_bot);
  seg_mask3 = splitted_HSV[1] <= (s_top);
  seg_mask4 = splitted_HSV[1] >= (s_bot);
  //handling cyclic range of hue

  /*cv::Mat seg_mask_t = (seg_mask1 > 0) & (seg_mask2 > 0) & (seg_mask3 > 0)
                          & (seg_mask4 > 0);*/
  cv::Mat seg_mask_t;
  if (h_top < h_bot){
    seg_mask_t = ((seg_mask1 > 0) | (seg_mask2 > 0)) & (seg_mask3 > 0)
                  & (seg_mask4 > 0);
  }
  else{
    seg_mask_t = (seg_mask1 > 0) & (seg_mask2 > 0) & (seg_mask3 > 0)
                 & (seg_mask4 > 0);
  }
  // mask
  /*cv::Mat seg_mask_t = (seg_mask1 > 0) & (seg_mask2 > 0) & (seg_mask3 > 0)
                         & (seg_mask4 > 0);*/
  cv::Mat seg_rgb_t = cv::Mat::zeros(src_rgb.size(), CV_8UC3);
  src_rgb.copyTo(seg_rgb_t, seg_mask_t);
  seg_mask = seg_mask_t;
  seg_rgb = seg_rgb_t;
}

//====================================================================================================================================

cv::Mat segmentation(cv::Mat src, int* hue_range, int* sat_range){
  //cv::Mat src_mb_rgb = src; // median burred src in rgb
  //[WARNING]: use same kernel size as in the calibration
  //cv::medianBlur(src, src_mb_rgb, 5);
  cv::Mat src_mb_hsv; // median blurred src hsv 
  cv::cvtColor(src, src_mb_hsv, CV_RGB2HSV);
  cv::Mat seg_mask, seg_out_rgb; 
  segmentHSV(src_mb_hsv, //in
             src, //in
             seg_mask, //out
             seg_out_rgb, //out
             hue_range[1], //in
             hue_range[0], //in
             sat_range[1], //in
             sat_range[0]); //in
  seg_mask = seg_mask / 255; // scale down to {0,1}
  return seg_mask;
}

//====================================================================================================================================

void segmentHSVEDIT(cv::Mat src, cv::Mat& seg_mask,
                    int h_top, int h_bot, int s_top, int s_bot){
  cv::Mat src_hsv;
  cv::cvtColor(src,src_hsv,CV_RGB2HSV);
  std::vector<cv::Mat> splitted_HSV;
  cv::split(src_hsv, splitted_HSV);
  cv::Mat seg_mask1 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  cv::Mat seg_mask2 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  cv::Mat seg_mask3 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  cv::Mat seg_mask4 = cv::Mat::zeros(src_hsv.size(), CV_8UC1);
  // Thresholding
  seg_mask1 = splitted_HSV[0] <= (h_top);
  seg_mask2 = splitted_HSV[0] >= (h_bot);
  seg_mask3 = splitted_HSV[1] <= (s_top);
  seg_mask4 = splitted_HSV[1] >= (s_bot);
  if (h_top < h_bot){
    seg_mask = ((seg_mask1 > 0) | (seg_mask2 > 0)) & (seg_mask3 > 0)
		& (seg_mask4 > 0);
  }
  else{
    seg_mask = (seg_mask1 > 0) & (seg_mask2 > 0) & (seg_mask3 > 0)
	        & (seg_mask4 > 0);
  }
  seg_mask = seg_mask / 255; //scale to {0,1}
}

//====================================================================================================================================

void noiseRemove(cv::Mat seg_mask, cv::Mat& seg_mask_noisefree, cv::Rect& box2){

  //[BOUNDING BOX]***********************************************
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(seg_mask, contours, hierarchy,
    CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  std::vector<std::vector<cv::Point> > contours_poly(contours.size());
  std::vector<cv::Rect> box(contours.size());
  double biggest_box = 0;
  int big1 = 0, big2 = 0; 
  for (int j = 0; j < (int)contours.size(); j++){
    cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, true);
    if (biggest_box < cv::contourArea(contours[j])){
      biggest_box = cv::contourArea(contours[j]);
      box[0] = cv::boundingRect(cv::Mat(contours_poly[j]));
      big1 = j;
    }
  }
  //***********************************************[BOUNDING BOX]

  //[REMOVE NOISE]***********************************************
  //cv::Mat tmp_img1 = cv::Mat::zeros(seg_mask.size(), CV_8UC1);
  //cv::Mat tmp_img2 = cv::Mat::zeros(seg_mask.size(), CV_8UC1);
  //seg_mask.rowRange(box[0].tl().y,box[0].br().y).copyTo(tmp_img1.rowRange(box[0].tl().y,box[0].br().y));
  //tmp_img1.colRange(box[0].tl().x,box[0].br().x).copyTo(tmp_img2.colRange(box[0].tl().x,box[0].br().x));
  //seg_mask_noisefree = tmp_img2;
  //***********************************************[REMOVE NOISE]

  cv::Mat tmp_img3 = cv::Mat::zeros(seg_mask.size(), CV_8UC1);
  cv::drawContours( tmp_img3, contours, big1, 1, -1);
  seg_mask_noisefree = tmp_img3;
  box2 = box[0];
}

//====================================================================================================================================


void noiseRemoveBox(cv::Mat seg_mask, cv::Mat& seg_mask_noisefree, cv::Rect& box2){

  //[BOUNDING BOX]***********************************************
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(seg_mask, contours, hierarchy,
    CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
  std::vector<std::vector<cv::Point> > contours_poly(contours.size());
  std::vector<cv::Rect> box(contours.size());
  double biggest_box = 0;
  int big1 = 0, big2 = 0; 
  for (int j = 0; j < (int)contours.size(); j++){
    cv::approxPolyDP(cv::Mat(contours[j]), contours_poly[j], 3, true);
    if (biggest_box < cv::contourArea(contours[j])){
      biggest_box = cv::contourArea(contours[j]);
      box[0] = cv::boundingRect(cv::Mat(contours_poly[j]));
      big1 = j;
    }
  }
  //***********************************************[BOUNDING BOX]

  cv::Mat tmp_img1 = cv::Mat::zeros(seg_mask.size(), CV_8UC1);
  cv::rectangle(tmp_img1,box[0].tl(),box[0].br(),1,-1);
  //cv::drawContours( tmp_img1, contours, big1, 1, -1);
  seg_mask_noisefree = tmp_img1;

  box2 = box[0];
}

//====================================================================================================================================


void pointCloudTrajectory(cv::Mat cloud, cv::Vec3f &point, std::vector<double> &EV){

  cv::Mat data_pts = cv::Mat(cloud.size().width*cloud.size().height, 3, CV_64FC1);

  cv::Vec3f single_point,tmp_point;
  cv::Vec3f pc_traj;
  float tmp = 5.0;
  int counter = 0;
  for(int i=0;i<cloud.size().height;i++){
  for(int ii=0;ii<cloud.size().width;ii++){
     single_point = cloud.at<cv::Vec3f>(i,ii);
     if(single_point[2]<tmp && single_point[2] > 0)
       tmp = single_point[2];
  }}
  for(int i=0;i<cloud.size().height;i++){
  for(int ii=0;ii<cloud.size().width;ii++){
    single_point = cloud.at<cv::Vec3f>(i,ii);
    if(single_point[2]<tmp+0.05 && single_point[2]>tmp-0.05){
      tmp_point += single_point; 
      data_pts.at<double>(counter, 0) = single_point[0];
      data_pts.at<double>(counter, 1) = single_point[1];
      data_pts.at<double>(counter, 2) = single_point[2];
      counter += 1;
    }
  }}
  tmp_point = tmp_point/counter;  
  pc_traj[0] = tmp_point[0];
  pc_traj[1] = tmp_point[1];
  pc_traj[2] = tmp_point[2];


  data_pts.resize(counter);
  //Perform PCA analysis
  cv::PCA pca_analysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);
  //Store the center of the object
  cv::Point3d cntr = cv::Point3d(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                                 static_cast<int>(pca_analysis.mean.at<double>(0, 1)),
                                 static_cast<int>(pca_analysis.mean.at<double>(0, 2)));
  //Store the eigenvalues and eigenvectors
  std::vector<cv::Point3d> eigen_vecs(3);
  std::vector<double> eigen_val(3);
  for (int i = 0; i < 3; ++i)
  {
      eigen_vecs[i] = cv::Point3d(pca_analysis.eigenvectors.at<double>(i, 0),
                                  pca_analysis.eigenvectors.at<double>(i, 1),
                                  pca_analysis.eigenvectors.at<double>(i, 2));
      eigen_val[i] = pca_analysis.eigenvalues.at<double>(0, i);
  }
  printf("EVs: %.4f %.4f %.4f    ",eigen_val[0]*1000,eigen_val[1]*1000,eigen_val[2]*1000);

  EV = eigen_val;
  point = pc_traj;
}


//====================================================================================================================================



void depthImaging(cv::Mat &depth_image, cv::Mat depth_global, uint16_t* mGamma){
    // Depth value processing
    u_char      *ptr=depth_image.data;
    uint16_t    *depth=(uint16_t*)depth_global.data;
    for(int i=0;i<640*480; ++i )
    {
      int pval = mGamma[depth[i]/2];
      int lb = pval & 0xff;
      switch ( pval >> 8 )
      {
          case 0:
            ptr[3*i+2] = 255;
            ptr[3*i+1] = 255-lb;
            ptr[3*i+0] = 255-lb;
            break;
          case 1:
            ptr[3*i+2] = 255;
            ptr[3*i+1] = lb;
            ptr[3*i+0] = 0;
          break;
        case 2:
          ptr[3*i+2] = 255-lb;
          ptr[3*i+1] = 255;
          ptr[3*i+0] = 0;
          break;
        case 3:
          ptr[3*i+2] = 0;
          ptr[3*i+1] = 255;
          ptr[3*i+0] = lb;
          break;
        case 4:
          ptr[3*i+2] = 0;
          ptr[3*i+1] = 255-lb;
          ptr[3*i+0] = 255;
          break;
        case 5:
          ptr[3*i+2] = 0;
          ptr[3*i+1] = 0;
          ptr[3*i+0] = 255-lb;
          break;
        default:
          ptr[3*i+2] = 0;
          ptr[3*i+1] = 0;
          ptr[3*i+0] = 0;
          break;
    }
  }
}

//====================================================================================================================================

std::vector<cv::Rect> detectFaceAndEyes( cv::Mat frame , cv::CascadeClassifier face_cascade){
  std::vector<cv::Rect> faces;
  cv::Mat frame_gray;
  cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
  //equalizeHist( frame_gray, frame_gray );
  //-- Detect faces
  face_cascade.detectMultiScale( frame_gray, faces, 1.2, 2, 0, cv::Size(60, 60), cv::Size(90, 90) );
  return faces;
}

//====================================================================================================================================

cv::Vec4f RANSAC3DPlane(cv::Mat cloud, cv::Mat &plane, int iter, float *ratio, float threshold){
  int i,x1,x2,x3,y1,y2,y3,x,y,counter,counter_max;
  double d_def,d_def_best,d_tmp;
  cv::Vec3f p1,p2,p3,p4,plane_norm,plane_best,p_check;

  counter_max = 0;
  srand(time(NULL));
  while (counter_max == 0){
  for (int i=0;i<iter;i++){         
    // random points in image plane (##### table lies below the mid line #####)
    x1 = rand() % 640; y1 = rand() % 480;
    x2 = rand() % 640; y2 = rand() % 480;
    x3 = rand() % 640; y3 = rand() % 480;
//    x1 = rand() % 640; y1 = (rand() % 240) +240;
//    x2 = rand() % 640; y2 = (rand() % 240) +240;
//    x3 = rand() % 640; y3 = (rand() % 240) +240;
    // prevent picking the same points
    while(x2==x1 && y2==y1) 
      {x2 = rand() % 640; y2 = rand() % 480;}       
    while(x3==x1 && y3==y1 && x3==x2 && y3==y2)   
      {x3 = rand() % 640; y3 = rand() % 480;}
    // random 3d points
    p1 = cloud.at<cv::Vec3f>(y1,x1);  
    p2 = cloud.at<cv::Vec3f>(y2,x2); 
    p3 = cloud.at<cv::Vec3f>(y3,x3); 
    p_check = crossProd(p1-p2,p2-p3); // prevent degenerate case
    if(p_check[0]!=0 && p_check[1]!=0 && p_check[2]!=0 && 
       p1[2]>0 && p2[2]>0 && p3[2]>0 && 
       p1[2]<2 && p2[2]<2 && p3[2]<2 )
    {
      counter = 0; 
      // hypothesis
      plane_norm = computePlane(p1, p2, p3);
      d_def = plane_norm[0]*p1[0]+plane_norm[1]*p1[1]+plane_norm[2]*p1[2];
      for(y=0;y<480;y++){
      for(x=0;x<640;x++){
        p4 = cloud.at<cv::Vec3f>(y,x); 
        d_tmp = plane_norm[0]*p4[0]+
                plane_norm[1]*p4[1]+
                plane_norm[2]*p4[2];   //offset plane from origin            
        if(abs(d_tmp-d_def)<threshold) counter +=1;              
      }}   
      if(counter<ratio[1]*(640*480) && 
         counter>ratio[0]*(640*480) && 
         counter>counter_max)
        {counter_max = counter; plane_best = plane_norm; d_def_best = d_def;}
    }
  }}
  
  // using the best points to build the mask
  counter = 0;
  for(y=0;y<480;y++){
  for(x=0;x<640;x++){
    p4 = cloud.at<cv::Vec3f>(y,x);               
    d_tmp = plane_best[0]*p4[0]+
            plane_best[1]*p4[1]+
            plane_best[2]*p4[2];   //offset plane from origin    
    if(abs(d_tmp-d_def_best)<threshold && p4[2]<2 && p4[2]>0){
      counter +=1;
      plane.data[(y*640)+x] = 1;         
    }             
  }}
  cv::Vec4f plane_constants;
  plane_constants[0] = plane_best[0];
  plane_constants[1] = plane_best[1];
  plane_constants[2] = plane_best[2];
  plane_constants[3] = d_def_best;
  return plane_constants;
}


//====================================================================================================================================

int markerContact(std::vector<cv::Vec3f> marker_center, cv::Vec4f plane, cv::Vec3f object_point){
  int contact_ = -1;
  float obj_plane, obj_marker, dist_diff;
  float dist_diff_min = 5.0;
  for (int i=0;i<marker_center.size();i++){
    obj_plane = object_point[0]*plane[0] + 
                object_point[1]*plane[1] + 
                object_point[2]*plane[2] - plane[3];
    obj_marker = ((object_point[0]-marker_center[i][0])*(object_point[0]-marker_center[i][0])) + 
                 ((object_point[1]-marker_center[i][1])*(object_point[1]-marker_center[i][1])) + 
                 ((object_point[2]-marker_center[i][2])*(object_point[2]-marker_center[i][2]));
    dist_diff = abs(obj_plane*obj_plane - obj_marker);
    //printf("  %.5f  ",dist_diff);
    if(dist_diff<0.005 && dist_diff<dist_diff_min) {contact_ = i; dist_diff_min = dist_diff;}
  }
  //std::cout << dist_diff << "\n"; fflush(stdout);
  return contact_;
}



//====================================================================================================================================

bool contactCheck(cv::Mat hand, cv::Rect object_blob){
  bool contact = false;
  if(object_blob.size().width>10 || object_blob.size().height>10){
    int hand_mask,y,x,y2,x2,tally,counter;
    counter = 0;
    for (y=0;y<480;y++){
    for (x=0;x<640;x++){
      tally = 0;
      if (hand.data[hand.cols*y + x]>0){	
        for (y2=y-15;y2<=y+15;y2++){
        for (x2=x-15;x2<=x+15;x2++){
	  if(x2>object_blob.x && 
             x2<object_blob.x+object_blob.size().width &&
             y2>object_blob.y &&
             y2<object_blob.y+object_blob.size().height)
             tally = 1;
        }}				
      }
      counter += tally;
    }}	
    if (counter > 10) contact = true;
  }
  return contact;
}


//====================================================================================================================================

float pointToSpeed(cv::Vec3f p1, cv::Vec3f p2){
  float speed;
  speed = cv::norm(p2-p1);
  return speed;
}

//====================================================================================================================================

cv::Vec3f pointToVelocity(cv::Vec3f p1, cv::Vec3f p2){
  cv::Vec3f velocity;
  velocity = p2-p1;
  return velocity;
}

//====================================================================================================================================

cv::Vec3f movingAveragePoint(std::vector<cv::Vec3f> points,float window){
  cv::Vec3f average(0,0,0);
  for (int i=0;i<points.size();i++)
    average += points[i];  
  average = average / window;
  return average;
}

//====================================================================================================================================

float movingAverageFloat(std::vector<float> data,float window){
  float average = 0.0;
  for (int i=0;i<data.size();i++)
    average += data[i];  
  average = average / window;
  return average;
}

//====================================================================================================================================

double movingAverageDouble(std::vector<double> data,float window){
  double average = 0.0;
  for (int i=0;i<data.size();i++)
    average += data[i];  
  average = average / window;
  return average;
}




