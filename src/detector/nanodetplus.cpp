// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <cfloat>
#include "cpu.h"
#include "platform.h"
#include "detector/nanodetplus.hpp"

static void generate_proposals(ncnn::Mat &cls_pred, ncnn::Mat &dis_pred, int stride, const ncnn::Mat &in_pad,
                               float prob_threshold, std::vector<DET_OBJ_T> &objects)
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
                    for (int l = 0; l < reg_max_1; l++) dis += l * dis_after_sm[l];
                    pred_ltrb[k] = dis * stride;
                }

                float x_center = j * stride;
                float y_center = i * stride;

                DET_OBJ_T obj;
                obj.x     = x_center - pred_ltrb[0];
                obj.y     = y_center - pred_ltrb[1];
                obj.w     = pred_ltrb[2] + pred_ltrb[0];
                obj.h     = pred_ltrb[3] + pred_ltrb[1];
                obj.label = max_label;
                obj.prob  = max_score;
                objects.push_back(obj);
            }
        }
    }
}

int NanoDetPlus::load(DET_PARAM_T *opt)
{
    if (!opt->stream_w || !opt->stream_h || !opt->model_path) return 0;

    char parampath[64];
    char modelpath[64];
    sprintf(parampath, "%s.param", opt->model_path);
    sprintf(modelpath, "%s.bin", opt->model_path);
    stream_w       = opt->stream_w;
    stream_h       = opt->stream_h;
    prob_threshold = opt->prob_threshold ? opt->prob_threshold : 0.4f;
    nms_threshold  = opt->nms_threshold ? opt->nms_threshold : 0.5f;

    target_size  = 416;
    mean_vals[0] = 103.53f;
    mean_vals[1] = 116.28f;
    mean_vals[2] = 123.675f;
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

int NanoDetPlus::detect(unsigned char *rgb, int width, int height, std::vector<DET_OBJ_T> &objects)
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
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
        in_pad.substract_mean_normalize(mean_vals, norm_vals);
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in_pad);
    std::vector<DET_OBJ_T> proposals;  // All objects
    // stride 8
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls8", cls_pred);
        ex.extract("dis8", dis_pred);

        std::vector<DET_OBJ_T> obj8;
        generate_proposals(cls_pred, dis_pred, 8, in_pad, prob_threshold, obj8);
        proposals.insert(proposals.end(), obj8.begin(), obj8.end());
    }

    // stride 16
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls16", cls_pred);
        ex.extract("dis16", dis_pred);

        std::vector<DET_OBJ_T> obj16;
        generate_proposals(cls_pred, dis_pred, 16, in_pad, prob_threshold, obj16);
        proposals.insert(proposals.end(), obj16.begin(), obj16.end());
    }

    // stride 32
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls32", cls_pred);
        ex.extract("dis32", dis_pred);

        std::vector<DET_OBJ_T> obj32;
        generate_proposals(cls_pred, dis_pred, 32, in_pad, prob_threshold, obj32);
        proposals.insert(proposals.end(), obj32.begin(), obj32.end());
    }

    // stride 64
    {
        ncnn::Mat cls_pred;
        ncnn::Mat dis_pred;
        ex.extract("cls64", cls_pred);
        ex.extract("dis64", dis_pred);

        std::vector<DET_OBJ_T> obj64;
        generate_proposals(cls_pred, dis_pred, 64, in_pad, prob_threshold, obj64);
        proposals.insert(proposals.end(), obj64.begin(), obj64.end());
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