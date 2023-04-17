cmake_minimum_required(VERSION 3.10)
project(DataEnhance)

set(OpenCV_DIR "C:\\opencv\\build\\x64\\mingw\\install")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories("C:\\Program Files (x86)\\Eigen3\\include")

add_executable(DataEnhance main.cpp)
target_link_libraries(DataEnhance ${OpenCV_LIBRARIES})