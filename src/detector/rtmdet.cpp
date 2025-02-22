#include <cfloat>
#include "cpu.h"
#include "platform.h"
#include "detector/rtmdet.hpp"

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

int RTMDet::load(DET_PARAM_T *opt)
{
    if (!opt->stream_w || !opt->stream_h || !opt->model_path) return 0;

    char parampath[64];
    char modelpath[64];
    sprintf(parampath, "%s.param", opt->model_path);
    sprintf(modelpath, "%s.bin", opt->model_path);

    stream_w       = opt->stream_w;
    stream_h       = opt->stream_h;
    prob_threshold = opt->prob_threshold ? opt->prob_threshold : 0.4f;
    nms_threshold  = opt->nms_threshold ? opt->nms_threshold : 0.6f;

    target_size  = 640;
    mean_vals[0] = 103.53;
    mean_vals[1] = 116.28;
    mean_vals[2] = 123.675;
    norm_vals[0] = 1.f / 57.375f;
    norm_vals[1] = 1.f / 57.12f;
    norm_vals[2] = 1.f / 58.395f;

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

static void generate_proposal(ncnn::Mat &cls, ncnn::Mat &dis, std::vector<DET_OBJ_T> &proposals, int stride,
                              float prob_thresh)
{
    const int feat_w   = cls.w;
    const int feat_h   = cls.h;
    const int feat_c   = cls.c;
    const int cls_step = cls.cstep;
    const int dis_step = dis.cstep;

    for (int i = 0; i < feat_h; i++) {
        for (int j = 0; j < feat_w; j++) {
            float *cls_ptr = cls.row(i) + j;
            float max_score = FLT_MIN;
            int   max_label = -1;
            for (int c = 0; c < feat_c; c++) {
                float score = FAST_SIGMOID(cls_ptr[c * cls_step]);
                if (score > max_score) {
                    max_score = score;
                    max_label = c;
                }
            }

            if (max_score > prob_thresh) {
                float *dis_ptr = dis.row(i) + j;

                DET_OBJ_T obj;
                obj.x     = j * stride - dis_ptr[0];
                obj.y     = i * stride - dis_ptr[dis_step];
                obj.w     = dis_ptr[dis_step * 2] + dis_ptr[0];
                obj.h     = dis_ptr[dis_step * 3] + dis_ptr[dis_step];
                obj.label = max_label;
                obj.prob  = max_score;
                proposals.push_back(obj);
            }
        }
    }
}

int RTMDet::detect(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects)
{
    // pad to multiple of 32
    int   w = width;
    int   h = height;
    float scale_rgb, scale_stream_w, scale_stream_h;
    if (w > h) {
        scale_rgb = (float)target_size / w;
        w         = target_size;
        h         = h * scale_rgb;
    } else {
        scale_rgb = (float)target_size / h;
        h         = target_size;
        w         = w * scale_rgb;
    }

    scale_stream_w = scale_rgb * stream_w / width;
    scale_stream_h = scale_rgb * stream_h / height;

    // Resize and make border
    int       wpad = (w + 31) / 32 * 32 - w;
    int       hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    {
        // We suspect that this weight was trained using BGR data, needs more confirmation
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
                               0.f);
        in_pad.substract_mean_normalize(mean_vals, norm_vals);
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("input", in_pad);
    std::vector<DET_OBJ_T> proposals;  // All objects
    // stride 8
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls8", cls_pred);
        ex.extract("dis8", dis_pred);

        std::vector<DET_OBJ_T> obj8;
        generate_proposal(cls_pred, dis_pred, obj8, 8, prob_threshold);
        proposals.insert(proposals.end(), obj8.begin(), obj8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls16", cls_pred);
        ex.extract("dis16", dis_pred);

        std::vector<DET_OBJ_T> obj16;
        generate_proposal(cls_pred, dis_pred, obj16, 16, prob_threshold);
        proposals.insert(proposals.end(), obj16.begin(), obj16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls32", cls_pred);
        ex.extract("dis32", dis_pred);

        std::vector<DET_OBJ_T> obj32;
        generate_proposal(cls_pred, dis_pred, obj32, 32, prob_threshold);
        proposals.insert(proposals.end(), obj32.begin(), obj32.end());
    }
    detncnn::qsort_descent_inplace(proposals);

    std::vector<int> picked;
    detncnn::nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].x - (wpad / 2)) / scale_stream_w;
        float y0 = (objects[i].y - (hpad / 2)) / scale_stream_h;
        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale_stream_w;
        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale_stream_h;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

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