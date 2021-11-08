cmake_minimum_required(VERSION 3.12)
add_compile_options(-std=c++17)
project(SwiftPR)

# options
option(DEBUG "Debug" OFF)

if(DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
else(DEBUG)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(include)

set(SRC_DETECTION src/PlateDetection.cpp  src/util.h include/PlateDetection.h)

set(SRC_FINEMAPPING src/FineMapping.cpp )

set(SRC_FASTDESKEW src/FastDeskew.cpp )

set(SRC_SEGMENTATION src/PlateSegmentation.cpp )

set(SRC_RECOGNIZE src/Recognizer.cpp src/CNNRecognizer.cpp)

set(SRC_PIPLINE src/Pipeline.cpp)

set(SRC_SEGMENTATIONFREE src/SegmentationFreeRecognizer.cpp )

#TEST_DETECTION
add_executable(test_detection ${SRC_DETECTION} demos/test_detection.cpp)
target_link_libraries(test_detection ${OpenCV_LIBS})

#TEST_FINEMAPPING
add_executable(test_finemapping ${SRC_FINEMAPPING} demos/test_finemapping.cpp)
target_link_libraries(test_finemapping ${OpenCV_LIBS})

#TEST_DESKEW

add_executable(TEST_FASTDESKEW ${SRC_FASTDESKEW} demos/test_fastdeskew.cpp)
target_link_libraries(TEST_FASTDESKEW ${OpenCV_LIBS})

#TEST_SEGMENTATION

add_executable(TEST_SEGMENTATION ${SRC_SEGMENTATION} ${SRC_RECOGNIZE} demos/test_segmentation.cpp)
target_link_libraries(TEST_SEGMENTATION ${OpenCV_LIBS})

#TEST_RECOGNIZATION

add_executable(TEST_RECOGNIZATION  ${SRC_RECOGNIZE} demos/test_recognization.cpp)
target_link_libraries(TEST_RECOGNIZATION ${OpenCV_LIBS})

#TEST_SEGMENTATIONFREE
add_executable(TEST_SEGMENTATIONFREE ${SRC_SEGMENTATIONFREE} demos/test_segmentationFree.cpp)
target_link_libraries(TEST_SEGMENTATIONFREE ${OpenCV_LIBS})

#TEST_PIPELINE

add_executable(TEST_PIPLINE ${SRC_DETECTION} ${SRC_FINEMAPPING} ${SRC_FASTDESKEW} ${SRC_SEGMENTATION} ${SRC_RECOGNIZE} ${SRC_PIPLINE} ${SRC_SEGMENTATIONFREE} demos/test_pipeline.cpp)
target_link_libraries(TEST_PIPLINE ${OpenCV_LIBS})

#test
set(TEST test)
add_executable(${TEST} demos/test.cpp ${SRC_DETECTION} ${SRC_FINEMAPPING} ${SRC_FASTDESKEW} ${SRC_SEGMENTATION} ${SRC_RECOGNIZE} ${SRC_PIPLINE} ${SRC_SEGMENTATIONFREE})
target_link_libraries(${TEST} ${OpenCV_LIBS})

#api
set(SRC_DIR src/)
set(API hyperlpr)
list(APPEND API_SRC api.cpp)
list(TRANSFORM API_SRC PREPEND ${SRC_DIR})
add_library(${API} SHARED ${API_SRC} ${SRC_DETECTION} ${SRC_FINEMAPPING} ${SRC_FASTDESKEW} ${SRC_SEGMENTATION} ${SRC_RECOGNIZE} ${SRC_PIPLINE} ${SRC_SEGMENTATIONFREE})
target_link_libraries(${API} ${OpenCV_LIBS})
