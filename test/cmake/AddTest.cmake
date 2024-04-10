function(add_test_c MODEL_NAME WEIGHT_NAME)
    configure_file(test_template.c.in ${WEIGHT_NAME}.c)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${WEIGHT_NAME}.bin ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHT_NAME}.bin COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/${WEIGHT_NAME}.param ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHT_NAME}.param COPYONLY)

    add_executable(test_${WEIGHT_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHT_NAME}.c)
    target_link_libraries(test_${WEIGHT_NAME} detkit)
    target_include_directories(test_${WEIGHT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()