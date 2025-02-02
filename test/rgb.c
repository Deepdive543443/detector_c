#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "detncnn.h"

static long long get_time()
{
    struct timeval te;
    gettimeofday(&te, NULL);
    long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;
    return milliseconds;
}

int main(int argc, char **argv)
{
    int            width, height, n;
    unsigned char *rgb = (unsigned char *)stbi_load(argv[1], &width, &height, &n, 3);
    if (rgb == NULL) {
        printf("Fail to load image %s\n", argv[1]);
        return 0;
    }

    // stbi_load() seems loading image in forms of BGR, so here comes a tiny convertion
    unsigned char *cvt_ptr = rgb;
    for (int i = 0; i < width * height; i++) {
        unsigned char r = cvt_ptr[0];
        unsigned char b = cvt_ptr[2];
        
        cvt_ptr[0] = b;
        cvt_ptr[2] = r;

        cvt_ptr += 3;
    }

    DET_PARAM_T opt = {.model_path = "nanodet-plus-m_416_int8", .stream_w = width, .stream_h = height};
    printf("width %d, height %d, n %d\n", width, height, n);

    void     *det     = nanodet_plus_init(&opt);
    DET_OBJ_T buf[DET_OBJ_BUFSIZE] = {};
    int out_len;
    int loops = atoi(argv[2]);
    for (int i = 0; i < loops; i++) 
    {
        long long timestamp = get_time();
        det_detect(det, rgb, height, width, buf, &out_len);
        timestamp = get_time() - timestamp;
        printf("[%4d/%4d] Duration: %lld ms\n", i + 1, loops,timestamp);
    }

    printf("%d obj detected\n", out_len);
    for (int i = 0; i < out_len; i++) {
        if (!buf[i].prob) {
            break;
        }
        printf("x: %f y: %f w: %f h: %f p: %f l: %d\n", buf[i].x, buf[i].y, buf[i].w, buf[i].h, buf[i].prob, buf[i].label);
    }

    stbi_image_free(rgb);
    det_exit(det);
    printf("Ja\n");
    return 0;
}