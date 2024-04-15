#include <cmath>
#include "detector.hpp"

namespace det {

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

class FastestDet : public Detector {
    int num_classes;

   public:
    FastestDet();

    virtual void load(int size, const char *path_param, const char *path_bin, int use_gpu);
    virtual void detect(cv::Mat &rgb, std::vector<Object> &objects) const;
};

FastestDet::FastestDet() {}

void FastestDet::load(int size, const char *path_param, const char *path_bin, int use_gpu)
{
#if NCNN_VULKAN
    network.opt.use_vulkan_compute = use_gpu;
#endif
    input_size = size;
    network.load_param(path_param);
    network.load_model(path_bin);

    mean_vals[0] = 0.f;
    mean_vals[1] = 0.f;
    mean_vals[2] = 0.f;

    norm_vals[0] = 0.00392157f;
    norm_vals[1] = 0.00392157f;
    norm_vals[2] = 0.00392157f;

    prob_thresh = 0.65;
    nms_thresh  = 0.65;
    num_classes = 80;
}

void FastestDet::detect(cv::Mat &rgb, std::vector<Object> &objects) const
{
    ncnn::Mat out;
    {
        ncnn::Mat input =
            ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR, rgb.cols, rgb.rows, input_size, input_size);

        input.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = network.create_extractor();

        ex.input("data", input);
        ex.extract("output", out);
    }

    int   c_step = out.cstep;
    float obj_score;

    std::vector<Object> proposals;

    for (int h = 0; h < out.h; h++) {
        float *w_ptr = out.row(h);
        for (int w = 0; w < out.w; w++) {
            float obj_score     = w_ptr[0];
            float max_cls_score = 0.0;
            int   max_cls_idx   = -1;

            for (int c = 0; c < num_classes; c++) {
                float cls_score = w_ptr[(c + 5) * c_step];
                if (cls_score > max_cls_score) {
                    max_cls_score = cls_score;
                    max_cls_idx   = c;
                }
            }

            if (pow(max_cls_score, 0.4) * pow(obj_score, 0.6) > 0.65) {
                float x_offset   = FAST_TANH(w_ptr[c_step]);
                float y_offset   = FAST_TANH(w_ptr[c_step * 2]);
                float box_width  = FAST_SIGMOID(w_ptr[c_step * 3]);
                float box_height = FAST_SIGMOID(w_ptr[c_step * 4]);
                float x_center   = (w + x_offset) / out.w;
                float y_center   = (h + y_offset) / out.h;

                Object obj;
                obj.rect.x      = (x_center - 0.5 * box_width) * rgb.cols;
                obj.rect.y      = (y_center - 0.5 * box_height) * rgb.rows;
                obj.rect.width  = box_width * rgb.cols;
                obj.rect.height = box_height * rgb.rows;
                obj.label       = max_cls_idx;
                obj.prob        = max_cls_score;
                proposals.push_back(obj);
            }
            w_ptr++;
        }
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms(proposals, picked, nms_thresh);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];
    }
    struct {
        bool operator()(const Object &a, const Object &b) const { return a.rect.area() > b.rect.area(); }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
}

Detector *create_fastestdet() { return new FastestDet; }
}  // namespace det