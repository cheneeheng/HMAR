cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

SET(GCC_GDB_COMPILE_FLAGS "-g")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_GDB_COMPILE_FLAGS}" )

include_directories( ~ /usr/local/src/XVision2/include /usr/local/intel/ipp/5.2/em64t/include )

link_directories( /usr/local/src/XVision2/lib /usr/local/intel/ipp/5.2/em64t/sharedlib ~/KINECT/images/color_select )

add_executable( kinect_tmp kinect_tmp.cpp )

target_link_libraries( kinect_tmp ${OpenCV_LIBS} XVSeg XVCons XVTrack XVDevs XVTools XVImages png Xext X11 ippccem64t ippcvem64t ippiem64t pthread)

