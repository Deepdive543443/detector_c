set(project_title detncnn)
project(${project_title})
cmake_minimum_required(VERSION 3.14)

include(${CMAKE_CURRENT_LIST_DIR}/test/AddTest.cmake)

if(DEFINED ENV{PROJ_TOP_DIR})
    set(ncnn_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/build/sdk/ncnn/lib/cmake/ncnn)
    find_package(ncnn REQUIRED)
else()
    set(ncnn_DIR ${CMAKE_CURRENT_LIST_DIR}/lib/build/pc/ncnn/lib/cmake/ncnn)
    find_package(ncnn REQUIRED)
endif()

file( GLOB TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/res/*)
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(name ${TEST_SOURCE} NAME)
    configure_file(${TEST_SOURCE} ${CMAKE_CURRENT_BINARY_DIR}/${name} COPYONLY)
endforeach()

file( GLOB_RECURSE LIB_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/*)
add_library( detncnn ${LIB_SOURCES})
target_include_directories(detncnn PRIVATE src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(detncnn ncnn)

install(TARGETS detncnn DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/lib)

add_test_c(nanodet-plus-m_416_int8 DET_NANODETPLUS)
add_test_c(FastestDet DET_FASTESTDET)
add_test_c(rtmdet_tiny_8xb32-300e_coco DET_RTMDET)
add_test_c(rtmdet_s_8xb32-300e_coco DET_RTMDET)

if(DEFINED ENV{PROJ_TOP_DIR})
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/lib/build/sdk/ncnn/lib/libncnn.a DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/lib)
else()
    install(FILES ${CMAKE_CURRENT_LIST_DIR}/lib/build/pc/ncnn/lib/libncnn.a DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/lib)
endif()
install(FILES ${CMAKE_CURRENT_LIST_DIR}/src/detncnn.h DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/include)