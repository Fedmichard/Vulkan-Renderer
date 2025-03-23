cmake_minimum_required(VERSION 3.26)
project(directx_11)

set(CMAKE_CXX_STANDARD 17)

include_directories(includes)

include_directories(libraries/glm)
include_directories(libraries/GLFW/include)
include_directories(C:/VulkanSDK/1.4.309.0/Include)

link_directories(libraries/GLFW/lib)
link_directories(C:/VulkanSDK/1.4.309.0/Lib)

add_executable(directx_11 src/main.cpp)

target_link_libraries(directx_11 libglfw3dll.a)
target_link_libraries(directx_11 vulkan-1.lib)
target_link_libraries(directx_11 user32.lib)
target_link_libraries(directx_11 gdi32.lib)
target_link_libraries(directx_11 shell32.lib)