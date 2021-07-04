cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)

SET(CMAKE_PREFIX_PATH /path/to/opencv)
SET(Torch_DIR /path/to/torch)
SET(CUDA_TOOLKIT_ROOT_DIR /path/to/cuda)


find_package(OpenCV REQUIRED ${OpenCV_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

if(NOT OpenCV_FOUND)
message(FATAL_ERROR "OpenCV Not Found!")
endif(NOT OpenCV_FOUND)

message(STATUS "OpenCV library status:")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")


find_package(Torch REQUIRED ${Torch_DIR})

if(NOT Torch_FOUND)
message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS " libraries: ${TORCH_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(seg_main seg_main.cpp)
target_link_libraries(seg_main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET seg_main PROPERTY CXX_STANDARD 14)
