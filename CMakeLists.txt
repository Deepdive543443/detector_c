cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

configure_file(${CMAKE_CURRENT_LIST_DIR}/include/detector.h ${CMAKE_CURRENT_BINARY_DIR}/detector.h COPYONLY)

set(ncnn_DIR ${NCNN_DIR})
find_package(ncnn REQUIRED)

configure_file(${CMAKE_CURRENT_LIST_DIR}/include/detector.h ${CMAKE_CURRENT_BINARY_DIR}/detector.h COPYONLY)

file( GLOB LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*.c)
add_library( detkit_c ${LIB_SOURCES})
target_include_directories(detkit_c PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(detkit_c ncnn)

install(TARGETS detkit_c DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/include/detector.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install)