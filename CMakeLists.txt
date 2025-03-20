cmake_minimum_required(VERSION 3.26)
project(directx_11)

set(CMAKE_CXX_STANDARD 17)

include_directories(includes)

add_executable(directx_11 src/main.cpp)