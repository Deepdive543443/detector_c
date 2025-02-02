#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "detncnn.h"

#define WIDTH  640
#define HEIGHT 360

static long long get_time()
{
    struct timeval te;
    gettimeofday(&te, NULL);
    long long milliseconds = te.tv_sec * 1000LL + te.tv_usec / 1000;
    return milliseconds;
}

int main(int argc, char **argv)
{
    DET_PARAM_T      opt  = {.model_path = "nanodet-plus-m_416_int8", .stream_w = WIDTH, .stream_h = HEIGHT};
    void          *det  = nanodet_plus_init(&opt);
    unsigned char *nv12 = malloc(WIDTH * HEIGHT * 3 / 2);
    int            nv12_size;

    FILE *bin = fopen(argv[1], "rb");
    if (bin) {
        fseek(bin, 0, SEEK_END);
        nv12_size = ftell(bin);
        fseek(bin, 0, SEEK_SET);

        if ((WIDTH * HEIGHT * 3 / 2) == nv12_size) {
            if (nv12) {
                fread(nv12, 1, nv12_size, bin);
                printf("Loaded\n");
            }
        }
        fclose(bin);
    }

    DET_OBJ_T buf[DET_OBJ_BUFSIZE] = {};
    int out_len;
    int loops = atoi(argv[2]);
    for (int i = 0; i < loops; i++) {
        long long timestamp = get_time();
        det_detect_nv12(det, nv12, HEIGHT, WIDTH, buf, &out_len);
        timestamp = get_time() - timestamp;
        printf("[%4d/%4d] Duration: %lld ms\n", i + 1, loops,timestamp);
    }

    printf("%d obj detected\n", out_len);
    for (int i = 0; i < out_len; i++) {
        if (!buf[i].prob) {
            break;
        }
        printf("x: %f y: %f w: %f h: %f p: %f l: %d\n", buf[i].x, buf[i].y, buf[i].w, buf[i].h, buf[i].prob,
               buf[i].label);
    }

    if (nv12) free(nv12);
    det_exit(det);
    printf("Ja\n");
    return 0;
}