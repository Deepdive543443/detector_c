cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

configure_file(${CMAKE_CURRENT_LIST_DIR}/include/detector.h ${CMAKE_CURRENT_BINARY_DIR}/detector.h COPYONLY)

option(BUILD_C "Build C API" OFF)
option(BUILD_CPP "Build CPP API" OFF)

set(ncnn_DIR ${NCNN_DIR})
find_package(ncnn REQUIRED)

if(BUILD_C)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/include/detector.h ${CMAKE_CURRENT_BINARY_DIR}/detector.h COPYONLY)

    file( GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.c)
    add_library( detkit_c ${LIB_SOURCES})
    target_include_directories(detkit_c PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(detkit_c ncnn)

    install(TARGETS detkit_c DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install)
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/include/detector.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install)
endif()

if(BUILD_CPP)
    set(OpenCV_DIR ${OCV_DIR})
    find_package(OpenCV REQUIRED core imgproc highgui)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/include/detector.hpp ${CMAKE_CURRENT_BINARY_DIR}/detector.hpp COPYONLY)

    file( GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
    add_library( detkit_cpp ${LIB_SOURCES})
    target_include_directories(detkit_cpp PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(detkit_cpp ncnn ${OpenCV_LIBS})

    install(TARGETS detkit_cpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install)
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/include/detector.hpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install)
endif()