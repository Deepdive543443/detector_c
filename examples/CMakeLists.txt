# RTSP streaming example (Vibed by Claude): cross-platform (macOS/Linux/Windows).
# Requires: OpenCV (videoio, imgproc), GStreamer >= 1.20 with gst-rtsp-server.
#
# Install hints:
#   macOS:   brew install opencv gstreamer
#   Ubuntu:  sudo apt install libopencv-dev libgstreamer1.0-dev \
#                            libgstreamer-plugins-base1.0-dev \
#                            libgstrtspserver-1.0-dev \
#                            gstreamer1.0-plugins-{base,good,bad,ugly} \
#                            gstreamer1.0-libav
#   Windows: install GStreamer MSVC dev + runtime from gstreamer.freedesktop.org,
#            install OpenCV, then configure with the GStreamer pkg-config dir
#            on PKG_CONFIG_PATH (or use -DCMAKE_PREFIX_PATH).

find_package(OpenCV QUIET COMPONENTS core imgproc videoio)
find_package(PkgConfig QUIET)

if(NOT OpenCV_FOUND OR NOT PkgConfig_FOUND)
    message(WARNING "rtsp_stream example skipped: OpenCV or pkg-config not found.")
    return()
endif()

pkg_check_modules(GST QUIET
    gstreamer-1.0
    gstreamer-app-1.0
    gstreamer-rtsp-server-1.0)

if(NOT GST_FOUND)
    message(WARNING "rtsp_stream example skipped: GStreamer / gst-rtsp-server not found via pkg-config.")
    return()
endif()

function(add_rtsp_example MODEL_NAME MODEL_TYPE)
    set(target rtsp_${MODEL_NAME})
    add_executable(${target} ${CMAKE_CURRENT_LIST_DIR}/rtsp_stream.cpp)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE
        RTSP_MODEL_TYPE=${MODEL_TYPE}
        RTSP_MODEL_NAME="${MODEL_NAME}")
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
        ${GST_INCLUDE_DIRS})
    target_link_directories(${target} PRIVATE ${GST_LIBRARY_DIRS})
    target_link_libraries(${target} PRIVATE
        detncnn
        ${OpenCV_LIBS}
        ${GST_LIBRARIES})
    target_compile_options(${target} PRIVATE ${GST_CFLAGS_OTHER})
    install(TARGETS ${target} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/install/bin)
endfunction()

add_rtsp_example(nanodet-plus-m_416_int8 DET_NANODETPLUS)
add_rtsp_example(FastestDet              DET_FASTESTDET)
add_rtsp_example(rtmdet_tiny_8xb32-300e_coco DET_RTMDET)
add_rtsp_example(rtmdet_s_8xb32-300e_coco    DET_RTMDET)
add_rtsp_example(rtmdet_m_8xb32-300e_coco    DET_RTMDET)
