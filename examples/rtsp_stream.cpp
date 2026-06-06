// Cross-platform RTSP object-detection streaming example vibed by Claude.
//
// Pipeline:
//   OpenCV VideoCapture (camera, cross-platform)
//     -> ncnn detector (existing det_* C API)
//     -> draw boxes
//     -> GStreamer appsrc -> H.264 encoder -> rtph264pay
//     -> gst-rtsp-server mount point  (default: rtsp://<host>:8554/stream)
//
// Tested platforms (encoder element auto-selected at runtime):
//   macOS   : vtenc_h264_hw  (VideoToolbox)
//   Linux   : x264enc        (software, always available)
//   Windows : mfh264enc      (Media Foundation) or x264enc fallback
//
// Build: enable -DBUILD_EXAMPLES=ON. Requires OpenCV, GStreamer >= 1.20
// with gst-rtsp-server, gst-plugins-{base,good,bad,ugly}.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "detector.hpp"

// -----------------------------------------------------------------------------
// Build-time defaults injected via CMake configure_file (see examples/CMakeLists.txt).
// -----------------------------------------------------------------------------
#ifndef RTSP_MODEL_TYPE
#define RTSP_MODEL_TYPE DET_NANODETPLUS
#endif
#ifndef RTSP_MODEL_NAME
#define RTSP_MODEL_NAME "nanodet-plus-m_416_int8"
#endif

// -----------------------------------------------------------------------------
// Per-platform GStreamer element selection.
// -----------------------------------------------------------------------------
static const char *pick_h264_encoder()
{
    static const char *candidates[] = {
#if defined(__APPLE__)
        "vtenc_h264_hw", "vtenc_h264",
#elif defined(_WIN32)
        "mfh264enc", "nvh264enc", "qsvh264enc",
#else
        "vaapih264enc", "nvh264enc",
#endif
        "x264enc", nullptr};
    for (int i = 0; candidates[i]; ++i) {
        GstElementFactory *f = gst_element_factory_find(candidates[i]);
        if (f) {
            gst_object_unref(f);
            return candidates[i];
        }
    }
    return "x264enc";
}

static std::string build_encoder_chain(const char *encoder)
{
    // Element-specific tuning for low latency.
    if (std::strcmp(encoder, "x264enc") == 0)
        return "x264enc tune=zerolatency speed-preset=superfast bitrate=2000 key-int-max=30";
    if (std::strncmp(encoder, "vtenc_h264", 10) == 0)
        return std::string(encoder) + " realtime=true allow-frame-reordering=false bitrate=2000";
    if (std::strcmp(encoder, "mfh264enc") == 0) return "mfh264enc bitrate=2000 rc-mode=cbr";
    if (std::strcmp(encoder, "nvh264enc") == 0) return "nvh264enc preset=low-latency-hq rc-mode=cbr-ld-hq bitrate=2000";
    if (std::strcmp(encoder, "vaapih264enc") == 0) return "vaapih264enc bitrate=2000 rate-control=cbr";
    if (std::strcmp(encoder, "qsvh264enc") == 0) return "qsvh264enc bitrate=2000";
    return encoder;  // default
}

// -----------------------------------------------------------------------------
// Shared state. Two workers:
//   1. capture_loop:  grabs frames at --fps, overlays last known boxes,
//                     pushes to appsrc. Also publishes the latest frame for
//                     the detector to consume.
//   2. detector_loop: takes a snapshot of the most recent frame, runs ncnn
//                     inference at --det-width (downscaled), publishes boxes
//                     scaled back to the streaming resolution.
// The detector never blocks the streaming pipeline — boxes lag by ~1 inference
// period but FPS stays stable.
// -----------------------------------------------------------------------------
struct StreamCtx {
    int                      width     = 640;
    int                      height    = 480;
    int                      fps       = 30;
    int                      cam_index = 0;
    int                      det_width = 416;  // longest side fed to detector
    Detector                *detector  = nullptr;
    GstClockTime             timestamp = 0;
    std::atomic<bool>        running{true};
    std::atomic<GstAppSrc *> appsrc{nullptr};
    cv::VideoCapture         cap;  // opened on main thread before worker starts

    // Latest captured frame, handed off to the detector thread.
    std::mutex              frame_mu;
    std::condition_variable frame_cv;
    cv::Mat                 latest_frame_bgr;  // protected by frame_mu
    uint64_t                frame_seq = 0;     // protected by frame_mu

    // Latest detection results, consumed by the streaming thread.
    std::mutex             dets_mu;
    std::vector<DET_OBJ_T> latest_dets;  // already in stream coords
};

// -----------------------------------------------------------------------------
// Detector thread: pull latest frame, downscale, run inference, publish boxes.
// Runs as fast as the model allows; never blocks capture/streaming.
// -----------------------------------------------------------------------------
static void detector_loop(StreamCtx *ctx)
{
    if (!ctx->detector) return;

    cv::Mat  small_bgr, small_rgb, frame_local;
    uint64_t last_seq = 0;

    while (ctx->running) {
        // Wait for a new frame.
        {
            std::unique_lock<std::mutex> lk(ctx->frame_mu);
            ctx->frame_cv.wait(lk, [&] { return !ctx->running || ctx->frame_seq != last_seq; });
            if (!ctx->running) break;
            ctx->latest_frame_bgr.copyTo(frame_local);
            last_seq = ctx->frame_seq;
        }
        if (frame_local.empty()) continue;

        // Downscale (longest side -> det_width) for fast inference.
        const int src_w     = frame_local.cols;
        const int src_h     = frame_local.rows;
        const int long_side = std::max(src_w, src_h);
        float     scale     = (long_side > ctx->det_width) ? static_cast<float>(ctx->det_width) / long_side : 1.0f;
        const int dw        = static_cast<int>(std::round(src_w * scale));
        const int dh        = static_cast<int>(std::round(src_h * scale));
        if (scale < 1.0f)
            cv::resize(frame_local, small_bgr, cv::Size(dw, dh));
        else
            small_bgr = frame_local;

        cv::cvtColor(small_bgr, small_rgb, cv::COLOR_BGR2RGB);

        std::vector<DET_OBJ_T> objs;
        ctx->detector->detect(small_rgb.data, small_rgb.cols, small_rgb.rows, objs);

        // Scale boxes back to the streaming resolution.
        if (scale < 1.0f) {
            const float inv = 1.0f / scale;
            for (auto &o : objs) {
                o.x *= inv;
                o.y *= inv;
                o.w *= inv;
                o.h *= inv;
            }
        }

        {
            std::lock_guard<std::mutex> lk(ctx->dets_mu);
            ctx->latest_dets = std::move(objs);
        }
    }
}

// -----------------------------------------------------------------------------
// Capture + push thread. Detection runs in its own thread; we just overlay the
// most recent results.
// -----------------------------------------------------------------------------
static void capture_loop(StreamCtx *ctx)
{
    const GstClockTime frame_dur = gst_util_uint64_scale_int(GST_SECOND, 1, ctx->fps);

    cv::Mat                frame_bgr, frame_rgb;
    std::vector<DET_OBJ_T> dets_snapshot;

    while (ctx->running) {
        if (!ctx->cap.read(frame_bgr) || frame_bgr.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (frame_bgr.cols != ctx->width || frame_bgr.rows != ctx->height)
            cv::resize(frame_bgr, frame_bgr, cv::Size(ctx->width, ctx->height));

        // Publish frame to detector thread (non-blocking, drops the previous one).
        if (ctx->detector) {
            std::lock_guard<std::mutex> lk(ctx->frame_mu);
            frame_bgr.copyTo(ctx->latest_frame_bgr);
            ++ctx->frame_seq;
            ctx->frame_cv.notify_one();

            // Snapshot most recent detections.
            std::lock_guard<std::mutex> dk(ctx->dets_mu);
            dets_snapshot = ctx->latest_dets;
        }

        if (!dets_snapshot.empty()) {
            cv::cvtColor(frame_bgr, frame_rgb, cv::COLOR_BGR2RGB);
            detncnn::draw_boxxes(frame_rgb.data, frame_rgb.cols, frame_rgb.rows, dets_snapshot);
            cv::cvtColor(frame_rgb, frame_bgr, cv::COLOR_RGB2BGR);
        }

        GstAppSrc *src = ctx->appsrc.load();
        if (!src) continue;  // no client connected yet

        const gsize size = static_cast<gsize>(frame_bgr.total() * frame_bgr.elemSize());
        GstBuffer  *buf  = gst_buffer_new_allocate(nullptr, size, nullptr);
        GstMapInfo  map;
        gst_buffer_map(buf, &map, GST_MAP_WRITE);
        std::memcpy(map.data, frame_bgr.data, size);
        gst_buffer_unmap(buf, &map);

        GST_BUFFER_PTS(buf)      = ctx->timestamp;
        GST_BUFFER_DTS(buf)      = ctx->timestamp;
        GST_BUFFER_DURATION(buf) = frame_dur;
        ctx->timestamp += frame_dur;

        GstFlowReturn ret = gst_app_src_push_buffer(src, buf);  // takes ownership
        if (ret != GST_FLOW_OK && ret != GST_FLOW_FLUSHING) {
            g_printerr("appsrc push returned %d\n", ret);
            break;
        }
    }
    ctx->cap.release();
    ctx->frame_cv.notify_all();  // wake detector so it can exit
}

// -----------------------------------------------------------------------------
// gst-rtsp-server: media-configure callback wires up the appsrc each time a
// client connects. We hand the appsrc to the capture thread via atomic pointer.
// -----------------------------------------------------------------------------
static void on_media_configure(GstRTSPMediaFactory * /*factory*/, GstRTSPMedia *media, gpointer user_data)
{
    auto       *ctx     = static_cast<StreamCtx *>(user_data);
    GstElement *element = gst_rtsp_media_get_element(media);
    GstElement *appsrc  = gst_bin_get_by_name_recurse_up(GST_BIN(element), "mysrc");
    if (!appsrc) {
        g_printerr("appsrc 'mysrc' not found in pipeline\n");
        gst_object_unref(element);
        return;
    }

    GstCaps *caps =
        gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "BGR", "width", G_TYPE_INT, ctx->width, "height",
                            G_TYPE_INT, ctx->height, "framerate", GST_TYPE_FRACTION, ctx->fps, 1, nullptr);
    g_object_set(appsrc, "caps", caps, "format", GST_FORMAT_TIME, "is-live", TRUE, "do-timestamp", FALSE, "block",
                 FALSE, "max-bytes", (guint64)0, "min-latency", (gint64)0, "max-latency", (gint64)0, nullptr);
    gst_caps_unref(caps);

    ctx->timestamp = 0;
    ctx->appsrc.store(GST_APP_SRC(appsrc));
    // Note: appsrc ref kept by pipeline; we don't unref here because we keep the pointer.
    gst_object_unref(element);
}

static void on_media_unprepared(GstRTSPMedia * /*media*/, gpointer user_data)
{
    auto *ctx = static_cast<StreamCtx *>(user_data);
    ctx->appsrc.store(nullptr);
}

static void on_media_configure_signals(GstRTSPMediaFactory *factory, GstRTSPMedia *media, gpointer user_data)
{
    on_media_configure(factory, media, user_data);
    g_signal_connect(media, "unprepared", G_CALLBACK(on_media_unprepared), user_data);
}

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------
static void usage(const char *prog)
{
    g_print(
        "Usage: %s [--cam N] [--width W] [--height H] [--fps F]\n"
        "          [--port 8554] [--mount /stream] [--model PATH]\n"
        "          [--det-width N]   detector input longest side (default 416)\n"
        "\n"
        "If --model is omitted, raw camera frames are streamed without inference.\n"
        "Detection runs in its own thread; stream FPS stays stable regardless of\n"
        "model speed (boxes lag video by ~1 inference period).\n"
        "RTSP URL: rtsp://<host>:<port><mount>\n",
        prog);
}

int main(int argc, char *argv[])
{
    // macOS: AVFoundation auth dialog can't be triggered from a worker thread.
    // Setting this skips the runtime request; user must pre-grant camera access
    // (System Settings → Privacy & Security → Camera → enable for Terminal/iTerm).
#if defined(__APPLE__)
    setenv("OPENCV_AVFOUNDATION_SKIP_AUTH", "1", 0);
#endif

    StreamCtx   ctx;
    int         port  = 8554;
    std::string mount = "/stream";
    std::string model_path;  // empty -> no detection

    for (int i = 1; i < argc; ++i) {
        std::string a    = argv[i];
        auto        next = [&](int &out) {
            if (i + 1 < argc) out = std::atoi(argv[++i]);
        };
        if (a == "--cam")
            next(ctx.cam_index);
        else if (a == "--width")
            next(ctx.width);
        else if (a == "--height")
            next(ctx.height);
        else if (a == "--fps")
            next(ctx.fps);
        else if (a == "--port")
            next(port);
        else if (a == "--mount" && i + 1 < argc)
            mount = argv[++i];
        else if (a == "--model" && i + 1 < argc)
            model_path = argv[++i];
        else if (a == "--det-width")
            next(ctx.det_width);
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else {
            g_printerr("Unknown arg: %s\n", a.c_str());
            usage(argv[0]);
            return 1;
        }
    }

    gst_init(&argc, &argv);

    // Strip trailing ".param" / ".bin" if user passed the full filename.
    // det_init expects the base path; it appends the extensions itself.
    for (const char *ext : {".param", ".bin"}) {
        const size_t n = std::strlen(ext);
        if (model_path.size() > n && model_path.compare(model_path.size() - n, n, ext) == 0) {
            model_path.resize(model_path.size() - n);
            break;
        }
    }

    // Open the camera on the MAIN thread. AVFoundation on macOS requires the
    // capture session to be created on the main run loop's thread; doing this
    // here also surfaces permission errors before we start the RTSP server.
    if (!ctx.cap.open(ctx.cam_index, cv::CAP_ANY)) {
        g_printerr(
            "Failed to open camera index %d.\n"
#if defined(__APPLE__)
            "On macOS: grant camera permission to your Terminal in\n"
            "  System Settings → Privacy & Security → Camera\n"
            "then re-run.\n",
#else
            "Check that the device exists and is not in use by another app.\n",
#endif
            ctx.cam_index);
        return 2;
    }
    ctx.cap.set(cv::CAP_PROP_FRAME_WIDTH, ctx.width);
    ctx.cap.set(cv::CAP_PROP_FRAME_HEIGHT, ctx.height);
    ctx.cap.set(cv::CAP_PROP_FPS, ctx.fps);
    ctx.width  = static_cast<int>(ctx.cap.get(cv::CAP_PROP_FRAME_WIDTH));
    ctx.height = static_cast<int>(ctx.cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    g_print("Camera opened: %dx%d @ %d fps\n", ctx.width, ctx.height, ctx.fps);

    if (!model_path.empty()) {
        // Compute the actual size we'll feed to the detector (longest side
        // limited by --det-width). stream_w/h must match the buffer the
        // detector will see, not the streaming resolution.
        const int long_side = std::max(ctx.width, ctx.height);
        float     scale     = (long_side > ctx.det_width) ? static_cast<float>(ctx.det_width) / long_side : 1.0f;
        const int det_w     = static_cast<int>(std::round(ctx.width * scale));
        const int det_h     = static_cast<int>(std::round(ctx.height * scale));

        DET_PARAM_T opt = {};
        opt.model_type  = RTSP_MODEL_TYPE;
        opt.model_path  = model_path.c_str();
        opt.stream_w    = det_w;
        opt.stream_h    = det_h;
        ctx.detector    = detncnn::init(&opt);
        if (!ctx.detector) {
            g_printerr("det_init failed (model: %s)\n", model_path.c_str());
            return 2;
        }
        g_print("Detection enabled (model: %s, inference at %dx%d)\n", model_path.c_str(), det_w, det_h);
    } else {
        g_print("No --model supplied: pure passthrough streaming.\n");
    }

    const char *enc       = pick_h264_encoder();
    std::string enc_chain = build_encoder_chain(enc);
    g_print("H.264 encoder: %s\n", enc);

    // Build the launch string for the RTSP media factory. appsrc is named "mysrc"
    // and located via gst_bin_get_by_name_recurse_up in on_media_configure().
    char launch[1024];
    std::snprintf(launch, sizeof(launch),
                  "( appsrc name=mysrc is-live=true do-timestamp=false format=time "
                  "  ! videoconvert ! video/x-raw,format=I420 "
                  "  ! %s "
                  "  ! h264parse config-interval=1 "
                  "  ! rtph264pay name=pay0 pt=96 config-interval=1 )",
                  enc_chain.c_str());

    GstRTSPServer *server = gst_rtsp_server_new();
    {
        char port_str[16];
        std::snprintf(port_str, sizeof(port_str), "%d", port);
        gst_rtsp_server_set_service(server, port_str);
    }
    GstRTSPMountPoints  *mounts  = gst_rtsp_server_get_mount_points(server);
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, launch);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_media_factory_set_latency(factory, 0);
    g_signal_connect(factory, "media-configure", G_CALLBACK(on_media_configure_signals), &ctx);
    gst_rtsp_mount_points_add_factory(mounts, mount.c_str(), factory);
    g_object_unref(mounts);

    if (gst_rtsp_server_attach(server, nullptr) == 0) {
        g_printerr("Failed to attach RTSP server (port busy?)\n");
        det_exit(ctx.detector);
        return 3;
    }

    g_print("RTSP stream ready at: rtsp://127.0.0.1:%d%s\n", port, mount.c_str());
    g_print("Test: ffplay -fflags nobuffer -flags low_delay rtsp://127.0.0.1:%d%s\n", port, mount.c_str());

    std::thread cap_thread(capture_loop, &ctx);
    std::thread det_thread;
    if (ctx.detector) det_thread = std::thread(detector_loop, &ctx);

    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(loop);

    ctx.running = false;
    ctx.frame_cv.notify_all();
    if (cap_thread.joinable()) cap_thread.join();
    if (det_thread.joinable()) det_thread.join();
    g_main_loop_unref(loop);
    if (ctx.detector) det_exit(ctx.detector);
    return 0;
}
