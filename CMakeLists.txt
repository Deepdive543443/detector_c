set(project_title detncnn)
project(${project_title})
cmake_minimum_required(VERSION 3.14)

# Configure ncnn build options
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "Disable ncnn build tools" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "Disable ncnn examples" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "Disable ncnn tests" FORCE)

file( GLOB_RECURSE LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*)
add_library( detncnn ${LIB_SOURCES})
target_include_directories(detncnn PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(detncnn ncnn)

add_subdirectory(lib/ncnn)
add_subdirectory(test)
