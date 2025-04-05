cmake_minimum_required(VERSION 3.26)
project(vukan-renderer)

set(CMAKE_CXX_STANDARD 17)

include_directories(includes)

include_directories(libraries/glm)
include_directories(libraries/GLFW/include)
include_directories(C:/VulkanSDK/1.4.309.0/Include)

link_directories(libraries/GLFW/lib)
link_directories(C:/VulkanSDK/1.4.309.0/Lib)

add_executable(vukan-renderer src/main.cpp)

target_link_libraries(vukan-renderer libglfw3dll.a)
target_link_libraries(vukan-renderer vulkan-1.lib)
target_link_libraries(vukan-renderer user32.lib)
target_link_libraries(vukan-renderer gdi32.lib)
target_link_libraries(vukan-renderer shell32.lib)