cmake_minimum_required(VERSION 3.0.0)
set (CMAKE_CXX_STANDARD 11)
project(detkit VERSION 0.1.0 LANGUAGES C CXX)

configure_file(${CMAKE_CURRENT_LIST_DIR}/include/detector.h ${CMAKE_CURRENT_BINARY_DIR}/detector.h COPYONLY)

file( GLOB LIB_SOURCES src/*.c)
add_library( detkit ${LIB_SOURCES})
target_include_directories(detkit PRIVATE ${CMAKE_CURRENT_BINARY_DIR})


set(ncnn_DIR ../ncnn_3ds_master/lib/cmake/ncnn) # Link ncnn
find_package(ncnn REQUIRED)
target_link_libraries(detkit ncnn)

install(TARGETS detkit DESTINATION ${CMAKE_CURRENT_LIST_DIR}/../build/detector)
install(FILES ${CMAKE_CURRENT_LIST_DIR}/include/detector.h DESTINATION ${CMAKE_CURRENT_LIST_DIR}/../build/detector)