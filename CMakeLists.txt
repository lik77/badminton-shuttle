cmake_minimum_required(VERSION 3.16)
project(shuttle)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# 海康 MVS SDK（仓库自带 x86/arm64 两套 .so）
set(HIK_ROOT ${CMAKE_SOURCE_DIR}/third_party/hikrobot)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(HIK_ARCH arm64)
else()
  set(HIK_ARCH amd64)
endif()
find_library(MVS_LIB libMvCameraControl.so PATHS ${HIK_ROOT}/lib/${HIK_ARCH} NO_DEFAULT_PATH)

# ONNX Runtime（YOLO 推理后端）
set(ORT_ROOT ${CMAKE_SOURCE_DIR}/third_party/onnxruntime)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS}
                    ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/src
                    ${HIK_ROOT}/include ${ORT_ROOT}/include)

add_executable(shuttle
  src/main.cpp
  src/shuttle_detector.cpp
  src/drag_predictor.cpp
  src/stereo.cpp
  src/rectifier.cpp
  src/yolo_proposer.cpp
  src/hik_camera.cpp
  src/stereo_camera.cpp
  src/rigs.cpp
  tools/extended_kalman_filter.cpp
)
target_link_libraries(shuttle ${OpenCV_LIBS} yaml-cpp ${MVS_LIB} pthread
                      ${ORT_ROOT}/lib/libonnxruntime.so)
set_target_properties(shuttle PROPERTIES
  BUILD_RPATH "${HIK_ROOT}/lib/${HIK_ARCH};${ORT_ROOT}/lib"
  INSTALL_RPATH "$ORIGIN/third_party/hikrobot/lib/${HIK_ARCH};$ORIGIN/third_party/onnxruntime/lib")
