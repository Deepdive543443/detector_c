#ifndef DETNCNN_H
#define DETNCNN_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    DET_NANODETPLUS,
    DET_FASTESTDET,
    DET_RTMDET,
} DET_MODEL_ENUM;

typedef struct {
    float x;
    float y;
    float w;
    float h;
    int   label;
    float prob;
} DET_OBJ_T;

typedef struct {
    DET_MODEL_ENUM model_type;
    const char    *model_path;
    int            stream_w;
    int            stream_h;
    float          prob_threshold;
    float          nms_threshold;
    int            use_gpu;
} DET_PARAM_T;

#define DET_OBJ_BUFSIZE 80

void *det_init(DET_PARAM_T *opt);
int   det_exit(void *ctx);
int   det_detect(void *ctx, unsigned char *rgb, int h, int w, DET_OBJ_T *output, int *out_len);
int   det_detect_nv12(void *ctx, unsigned char *nv12, int h, int w, DET_OBJ_T *output, int *out_len);
int   det_draw_boxxes(unsigned char *rgb, int h, int w, DET_OBJ_T *output, int *out_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // DETNCNN_H