function(add_test_c MODEL_NAME WEIGHT_NAME INPUT_SIZE)
    configure_file(test_template.c.in ${WEIGHT_NAME}.c)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/weight/${WEIGHT_NAME}.bin ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHT_NAME}.bin COPYONLY)
    configure_file(${CMAKE_CURRENT_LIST_DIR}/weight/${WEIGHT_NAME}.param ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHT_NAME}.param COPYONLY)

    add_executable(test_${WEIGHT_NAME}_c ${CMAKE_CURRENT_BINARY_DIR}/${WEIGHT_NAME}.c)
    target_link_libraries(test_${WEIGHT_NAME}_c detkit)
    target_include_directories(test_${WEIGHT_NAME}_c PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endfunction()