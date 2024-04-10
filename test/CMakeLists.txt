cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/AddTest.cmake)

configure_file(${CMAKE_CURRENT_LIST_DIR}/../include/detector.h ${CMAKE_CURRENT_BINARY_DIR}/detector.h COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/../include/stb_image.h ${CMAKE_CURRENT_BINARY_DIR}/stb_image.h COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/../include/stb_image_write.h ${CMAKE_CURRENT_BINARY_DIR}/stb_image_write.h COPYONLY)

configure_file(${CMAKE_CURRENT_LIST_DIR}/test1.jpg ${CMAKE_CURRENT_BINARY_DIR}/test1.jpg COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/test2.jpg ${CMAKE_CURRENT_BINARY_DIR}/test2.jpg COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/test3.jpg ${CMAKE_CURRENT_BINARY_DIR}/test3.jpg COPYONLY)

file( GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/../src/*.c)
add_library( detkit ${LIB_SOURCES})
target_include_directories(detkit PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

set(ncnn_DIR ${NCNN_DIR})
find_package(ncnn REQUIRED)
target_link_libraries(detkit ncnn)

add_test_c(nanodet nanodet-plus-m_416_int8)
add_test_c(fastestdet FastestDet)