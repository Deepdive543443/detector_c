function(add_test_c CMAKE_MODEL_NAME CMAKE_INIT_FUNC)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/test/rgb.c.in ${CMAKE_CURRENT_BINARY_DIR}/rgb_${CMAKE_MODEL_NAME}.c)

    add_executable(rgb_${CMAKE_MODEL_NAME} ${CMAKE_CURRENT_BINARY_DIR}/rgb_${CMAKE_MODEL_NAME}.c)
    target_include_directories(rgb_${CMAKE_MODEL_NAME} PRIVATE src lib ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(rgb_${CMAKE_MODEL_NAME} detncnn)

    install(TARGETS rgb_${CMAKE_MODEL_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/bin)

    configure_file(${CMAKE_CURRENT_LIST_DIR}/test/nv12.c.in ${CMAKE_CURRENT_BINARY_DIR}/nv12_${CMAKE_MODEL_NAME}.c)

    add_executable(nv12_${CMAKE_MODEL_NAME} ${CMAKE_CURRENT_BINARY_DIR}/nv12_${CMAKE_MODEL_NAME}.c)
    target_include_directories(nv12_${CMAKE_MODEL_NAME} PRIVATE src lib ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(nv12_${CMAKE_MODEL_NAME} detncnn)

    install(TARGETS nv12_${CMAKE_MODEL_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/bin)
endfunction()