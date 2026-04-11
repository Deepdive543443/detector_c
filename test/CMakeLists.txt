function(add_test_c CMAKE_MODEL_NAME CMAKE_MODEL_TYPE)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/rgb.c.in ${CMAKE_CURRENT_BINARY_DIR}/rgb_${CMAKE_MODEL_NAME}.c)

    add_executable(rgb_${CMAKE_MODEL_NAME} ${CMAKE_CURRENT_BINARY_DIR}/rgb_${CMAKE_MODEL_NAME}.c)
    target_include_directories(rgb_${CMAKE_MODEL_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(rgb_${CMAKE_MODEL_NAME} detncnn)

    install(TARGETS rgb_${CMAKE_MODEL_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/bin)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/nv12.c.in ${CMAKE_CURRENT_BINARY_DIR}/nv12_${CMAKE_MODEL_NAME}.c)

    add_executable(nv12_${CMAKE_MODEL_NAME} ${CMAKE_CURRENT_BINARY_DIR}/nv12_${CMAKE_MODEL_NAME}.c)
    target_include_directories(nv12_${CMAKE_MODEL_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/lib ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(nv12_${CMAKE_MODEL_NAME} detncnn)

    install(TARGETS nv12_${CMAKE_MODEL_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/bin)
endfunction()

add_test_c(nanodet-plus-m_416_int8 DET_NANODETPLUS)
add_test_c(FastestDet DET_FASTESTDET)
add_test_c(rtmdet_tiny_8xb32-300e_coco DET_RTMDET)
add_test_c(rtmdet_s_8xb32-300e_coco DET_RTMDET)
add_test_c(rtmdet_m_8xb32-300e_coco DET_RTMDET)

file( GLOB TEST_SOURCES ${CMAKE_CURRENT_LIST_DIR}/res/*)
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(name ${TEST_SOURCE} NAME)
    configure_file(${TEST_SOURCE} ${CMAKE_CURRENT_BINARY_DIR}/${name} COPYONLY)
endforeach()