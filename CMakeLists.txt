cmake_minimum_required(VERSION 3.25)
project(untitled1)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_FIND_DEBUG_MODE TRUE)
find_package (Eigen3  REQUIRED NO_MODULE)
find_package(autodiff REQUIRED)



add_executable(untitled1 main.cpp)
target_link_libraries(untitled1 PUBLIC autodiff::autodiff Eigen3::Eigen)