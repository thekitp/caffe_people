cmake_minimum_required(VERSION 3.3)
project(Caffe_People)

find_package(Caffe REQUIRED)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

option(CPU_ONLY "Use CPU only for Caffe" OFF)
option(USE_CUDNN "Use CuDNN library for Caffe" ON)

find_package(OpenCV REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.55.0 COMPONENTS filesystem)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/home/aurora/hdd/workspace/ClionProjects2/caffe_new/caffe-master/include)
include_directories(/home/aurora/hdd/workspace/ClionProjects2/caffe_new/caffe-master/.build_release/src)
#include_directories(/home/aurora/software/jdk1.8.0_25/include)
#include_directories(/home/aurora/software/jdk1.8.0_25/include/linux)

include_directories(${PROJECT_SOURCE_DIR})

include_directories("/home/aurora/hdd/software/HSKW/incCn/")


SET ( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g" )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(Caffe_DIR "/home/aurora/workspace/ClionProjects/caffe2/build/")

set(PROJECTNAME Caffe_People)
#aux_source_directory(. SOURCE_FILES)
#add_executable(Caffe_People get_features.cpp)
#add_executable(Caffe_People get_features.cpp)
#add_executable(Caffe_People test_pic_1.cpp)
add_executable(Caffe_People test_new_layer.cpp)
#add_executable(Caffe_People tcp_test.cpp)
#add_executable(Caffe_People read_video.cpp read_video2.cpp)
#add_executable(Caffe_People read_video2.cpp)

target_link_libraries(${PROJECTNAME} /usr/lib/x86_64-linux-gnu/libglog.so)
target_link_libraries(${PROJECTNAME} /usr/lib/x86_64-linux-gnu/libboost_system.so)
#target_link_libraries(Caffe_People ${Caffe_LIBRARIES})
target_link_libraries(${PROJECTNAME} ${OpenCV_LIBRARIES})
#target_link_libraries(${PROJECTNAME} ${Caffe_LIBRARIES})
target_link_libraries(${PROJECTNAME} ${Boost_LIBRARIES})
target_link_libraries(${PROJECTNAME} /home/aurora/hdd/workspace/ClionProjects2/caffe_new/caffe-master/build/lib/libcaffe.so)

#target_link_libraries(${PROJECTNAME} /home/aurora/software/jdk1.8.0_25/jre/lib/amd64/serve/libjvm.so)
#target_link_libraries(${PROJECTNAME} /home/aurora/hdd/software/HSKW/lib/libPlayCtrl.so)
#target_link_libraries(${PROJECTNAME} /home/aurora/hdd/software/HSKW/lib/libAudioRender.so)
#target_link_libraries(${PROJECTNAME} /home/aurora/hdd/software/HSKW/lib/libSuperRender.so)
target_link_libraries(${PROJECTNAME} /home/aurora/hdd/software/HSKW/lib/libHCCore.so)
target_link_libraries(${PROJECTNAME} /home/aurora/hdd/software/HSKW/lib/libhcnetsdk.so)
target_link_libraries(${PROJECTNAME} /home/aurora/hdd/software/HSKW/lib/libhpr.so)
target_link_libraries(${PROJECTNAME} /opt/intel/mkl/lib/intel64/libmkl_rt.so)
target_link_libraries(${PROJECTNAME} "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libanalyzedata.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCAlarm.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCCoreDevCfg.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCDisplay.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCGeneralCfgMgr.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCIndustry.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCPlayBack.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCPreview.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libHCVoiceTalk.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libiconv2.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libStreamTransClient.so"
        "/home/aurora/hdd/software/HSKW/lib/HCNetSDKCom/libSystemTransform.so")