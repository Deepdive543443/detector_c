#include <cfloat>
#include "cpu.h"
#include "platform.h"
#include "detector/fastestdet.hpp"

static inline float fast_exp(float x)
{
    union {
        __uint32_t i;
        float      f;
    } v;
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
#define FAST_SIGMOID(X) (1.0f / (1.0f + fast_exp(-X)))
#define FAST_TANH(X)    (2.f / (1.f + fast_exp(-2 * X)) - 1.f)

int FastestDet::load(DET_PARAM_T *opt)
{
    if (!opt->stream_w || !opt->stream_h || !opt->model_path) return 0;

    char parampath[64];
    char modelpath[64];
    sprintf(parampath, "%s.param", opt->model_path);
    sprintf(modelpath, "%s.bin", opt->model_path);

    stream_w       = opt->stream_w;
    stream_h       = opt->stream_h;
    prob_threshold = opt->prob_threshold ? opt->prob_threshold : 0.65f;
    nms_threshold  = opt->nms_threshold ? opt->nms_threshold : 0.65f;

    target_size  = 352;
    mean_vals[0] = 0.0f;
    mean_vals[1] = 0.0f;
    mean_vals[2] = 0.0f;
    norm_vals[0] = 0.00392157f;
    norm_vals[1] = 0.00392157f;
    norm_vals[2] = 0.00392157f;

    net.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    net.opt = ncnn::Option();

#if NCNN_VULKAN
    net.opt.use_vulkan_compute = opt->use_gpu;
#endif

    net.opt.num_threads         = ncnn::get_big_cpu_count();
    net.opt.blob_allocator      = &blob_pool_allocator;
    net.opt.workspace_allocator = &workspace_pool_allocator;

    net.load_param(parampath);
    net.load_model(modelpath);
    return 1;
}

int FastestDet::detect(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects)
{
    ncnn::Mat in =
        ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB2BGR, width, height, target_size, target_size);
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in);

    ncnn::Mat out;
    ex.extract("output", out);

    const int out_c_step = out.cstep;
    const int out_h      = out.h;
    const int out_w      = out.w;

    std::vector<DET_OBJ_T> proposals;
    for (int h = 0; h < out_h; h++) {
        float *w_ptr = out.row(h);
        for (int w = 0; w < out_w; w++) {
            float obj_score     = w_ptr[0];
            float max_cls_score = 0.f;
            int   max_cls_idx   = -1;

            for (int c = 0; c < 80; c++) {
                float cls_score = w_ptr[(c + 5) * out_c_step];
                if (cls_score > max_cls_score) {
                    max_cls_score = cls_score;
                    max_cls_idx   = c;
                }
            }

            if (pow(max_cls_score, 0.4) * pow(obj_score, 0.6) > prob_threshold) {
                float x_offset   = FAST_TANH(w_ptr[out_c_step]);
                float y_offset   = FAST_TANH(w_ptr[out_c_step * 2]);
                float box_width  = FAST_SIGMOID(w_ptr[out_c_step * 3]);
                float box_height = FAST_SIGMOID(w_ptr[out_c_step * 4]);
                float x_center   = (w + x_offset) / out_w;
                float y_center   = (h + y_offset) / out_h;

                DET_OBJ_T info;
                info.x     = x_center - 0.5 * box_width;
                info.y     = y_center - 0.5 * box_height;
                info.w     = box_width;
                info.h     = box_height;
                info.label = max_cls_idx;
                info.prob  = obj_score;
                proposals.push_back(info);
            }
            w_ptr++;
        }
    }
    detncnn::qsort_descent_inplace(proposals);

    std::vector<int> picked;
    detncnn::nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = objects[i].x * stream_w;
        float y0 = objects[i].y * stream_h;
        float x1 = (objects[i].x + objects[i].w) * stream_w;
        float y1 = (objects[i].y + objects[i].h) * stream_h;

        // clip
        x0 = std::max(std::min(x0, (float)(stream_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(stream_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(stream_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(stream_h - 1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].w = x1 - x0;
        objects[i].h = y1 - y0;
    }
    struct {
        bool operator()(const DET_OBJ_T &a, const DET_OBJ_T &b) const { return (a.w * a.h) > (b.w * b.h); }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
    return 1;
}
