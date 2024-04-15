#include "detector.hpp"

namespace det {

class NanodetPlus : public Detector {
   public:
    NanodetPlus();

    virtual void load(int size, const char *path_param, const char *path_bin, int use_gpu);
    virtual void detect(cv::Mat &rgb, std::vector<Object> &objects) const;
};

static void generate_proposals(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride, const ncnn::Mat &in_pad,
                               float prob_threshold, std::vector<Object> &objects)
{
    const int num_grid_x = cls_pred.w;
    const int num_grid_y = cls_pred.h;
    const int num_class  = cls_pred.c;
    const int cstep_cls  = cls_pred.cstep;

    const int reg_max_1 = dis_pred.w / 4;
    const int hstep_dis = dis_pred.cstep;

    for (int i = 0; i < num_grid_y; i++) {
        for (int j = 0; j < num_grid_x; j++) {
            float *score_ptr = cls_pred.row(i) + j;
            float  max_score = -FLT_MAX;
            int    max_label = -1;

            for (int cls = 0; cls < num_class; cls++) {
                if (score_ptr[cls * cstep_cls] > max_score) {
                    max_score = score_ptr[cls * cstep_cls];
                    max_label = cls;
                }
            }

            if (max_score >= prob_threshold) {
                ncnn::Mat bbox_pred(reg_max_1, 4, (void *)(dis_pred.row(j) + i * hstep_dis));
                {
                    ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                    ncnn::ParamDict pd;
                    pd.set(0, 1);  // axis
                    pd.set(1, 1);
                    softmax->load_param(pd);

                    ncnn::Option opt;
                    opt.num_threads        = 1;
                    opt.use_packing_layout = false;

                    softmax->create_pipeline(opt);
                    softmax->forward_inplace(bbox_pred, opt);
                    softmax->destroy_pipeline(opt);

                    delete softmax;
                }

                float pred_ltrb[4];
                for (int k = 0; k < 4; k++) {
                    float        dis          = 0.f;
                    const float *dis_after_sm = bbox_pred.row(k);
                    for (int l = 0; l < reg_max_1; l++) {
                        dis += l * dis_after_sm[l];
                    }
                    pred_ltrb[k] = dis * stride;
                }

                float x_center = j * stride;
                float y_center = i * stride;

                Object obj;
                obj.rect.x      = x_center - pred_ltrb[0];
                obj.rect.y      = y_center - pred_ltrb[1];
                obj.rect.width  = pred_ltrb[2] + pred_ltrb[0];
                obj.rect.height = pred_ltrb[3] + pred_ltrb[1];
                obj.label       = max_label;
                obj.prob        = max_score;
                objects.push_back(obj);
            }
        }
    }
}

NanodetPlus::NanodetPlus()
{
    network.opt.use_int8_inference = true;

    mean_vals[0] = 103.53f;
    mean_vals[1] = 116.28f;
    mean_vals[2] = 123.675f;

    norm_vals[0] = 1.f / 57.375f;
    norm_vals[1] = 1.f / 57.12f;
    norm_vals[2] = 1.f / 58.395f;

    prob_thresh = 0.4;
    nms_thresh  = 0.5;
}

void NanodetPlus::load(int size, const char *path_param, const char *path_bin, int use_gpu)
{
#if NCNN_VULKAN
    network.opt.use_vulkan_compute = use_gpu;
#endif
    network.opt.use_int8_inference = true;
    input_size = size;
    network.load_param(path_param);
    network.load_model(path_bin);
}

void NanodetPlus::detect(cv::Mat &rgb, std::vector<Object> &objects) const
{
    const int pixel_w = rgb.cols;
    const int pixel_h = rgb.rows;

    int   w, h;
    float scale;
    if (pixel_w > pixel_h) {
        scale = (float)input_size / pixel_w;
        w     = input_size;
        h     = pixel_h * scale;
    } else {
        scale = (float)input_size / pixel_h;
        h     = input_size;
        w     = pixel_w * scale;
    }

    const int wpad = (w + 31) / 32 * 32 - w;
    const int hpad = (h + 31) / 32 * 32 - h;

    ncnn::Mat in_pad;
    {
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, pixel_w, pixel_h, w, h);

        int wpad = (w + 31) / 32 * 32 - w;
        int hpad = (h + 31) / 32 * 32 - h;
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT,
                               0.f);
    }

    in_pad.substract_mean_normalize(mean_vals, norm_vals);
    ncnn::Extractor ex = network.create_extractor();
    ex.input("data", in_pad);

    std::vector<Object> proposals;

    const char *outputs[] = {"dis8", "cls8", "dis16", "cls16", "dis32", "cls32", "dis64", "cls64"};
    const int   strides[] = {8, 16, 32, 64};
    for (int i = 0; i < 4; i++) {
        ncnn::Mat dis_pred, cls_pred;
        ex.extract(outputs[i * 2], dis_pred);
        ex.extract(outputs[i * 2 + 1], cls_pred);

        std::vector<Object> obj_scale;
        generate_proposals(cls_pred, dis_pred, strides[i], in_pad, prob_thresh, obj_scale);
        proposals.insert(proposals.end(), obj_scale.begin(), obj_scale.end());
    }

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms(proposals, picked, nms_thresh);

    int count = picked.size();
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(pixel_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(pixel_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(pixel_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(pixel_h - 1)), 0.f);

        objects[i].rect.x      = x0;
        objects[i].rect.y      = y0;
        objects[i].rect.width  = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
    struct {
        bool operator()(const Object &a, const Object &b) const { return a.rect.area() > b.rect.area(); }
    } objects_area_greater;
    std::sort(objects.begin(), objects.end(), objects_area_greater);
};

Detector *create_nanodet_plus() { return new NanodetPlus; }
}  // namespace det
