# cmake -DNCNN_DIR=/home/qf28/cpp_projects/ncnn/build/install/lib/cmake/ncnn -DOCV_DIR=/home/qfeng10/opencv-mobile-4.9.0/build/install/lib/cmake/opencv4 -DTEST_CPP=ON ../test 
cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/AddTest.cmake)

option(TEST_C "Build C API test" ON)
option(TEST_CPP "Build CPP test" OFF)

set(ncnn_DIR ${NCNN_DIR})
find_package(ncnn REQUIRED)

if(TEST_C)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../include/detector.h ${CMAKE_CURRENT_BINARY_DIR}/detector.h COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../include/stb_image.h ${CMAKE_CURRENT_BINARY_DIR}/stb_image.h COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/../include/stb_image_write.h ${CMAKE_CURRENT_BINARY_DIR}/stb_image_write.h COPYONLY)

    file( GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/../src/*.c)
    add_library( detkit ${LIB_SOURCES})
    target_include_directories(detkit PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(detkit ncnn)

    add_test_c(nanodet_plus nanodet-plus-m_416_int8 416)
    add_test_c(fastestdet FastestDet 352)
endif()

if(TEST_CPP)
    set(OpenCV_DIR ${OCV_DIR})
    find_package(OpenCV REQUIRED core imgproc highgui)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/../include/detector.hpp ${CMAKE_CURRENT_BINARY_DIR}/detector.hpp COPYONLY)

    file( GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/../src/*.cpp)
    add_library( detkit_cpp ${LIB_SOURCES})
    target_include_directories(detkit_cpp PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(detkit_cpp ncnn ${OpenCV_LIBS})

    add_test_cpp(nanodet_plus nanodet-plus-m_416_int8 416)
    add_test_cpp(fastestdet FastestDet 352)
endif()


configure_file(${CMAKE_CURRENT_LIST_DIR}/img/test1.jpg ${CMAKE_CURRENT_BINARY_DIR}/test1.jpg COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/img/test2.jpg ${CMAKE_CURRENT_BINARY_DIR}/test2.jpg COPYONLY)
configure_file(${CMAKE_CURRENT_LIST_DIR}/img/test3.jpg ${CMAKE_CURRENT_BINARY_DIR}/test3.jpg COPYONLY)