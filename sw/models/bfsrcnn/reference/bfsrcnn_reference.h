/* Model-side integer golden reference API. */
#ifndef BFSRCNN_HW_INFERENCE_H
#define BFSRCNN_HW_INFERENCE_H

#include <stdint.h>

typedef enum {
    BFSRCNN_HW_STAGE_HEAD = 0,
    BFSRCNN_HW_STAGE_SHRINK1,
    BFSRCNN_HW_STAGE_SHRINK2,
    BFSRCNN_HW_STAGE_SHRINK3,
    BFSRCNN_HW_STAGE_MAPPING0,
    BFSRCNN_HW_STAGE_MAPPING1,
    BFSRCNN_HW_STAGE_MAPPING2,
    BFSRCNN_HW_STAGE_MAPPING3,
    BFSRCNN_HW_STAGE_MAPPING4,
    BFSRCNN_HW_STAGE_MAPPING5,
    BFSRCNN_HW_STAGE_MAPPING6,
    BFSRCNN_HW_STAGE_MAPPING7,
    BFSRCNN_HW_STAGE_EXPAND,
    BFSRCNN_HW_STAGE_COUNT
} bfsrcnn_hw_stage_t;

typedef void (*bfsrcnn_hw_stage_callback_t)(
    bfsrcnn_hw_stage_t stage,
    const int32_t *data,
    int height,
    int width,
    int channels,
    void *user);

/*
 * Input and output are single-channel, row-major uint8 images.
 * Returns 0 on success and -1 if temporary buffer allocation fails.
 */
int bfsrcnn_hw_infer(
    const uint8_t *input,
    int input_height,
    int input_width,
    uint8_t *output
);

int bfsrcnn_hw_infer_stages(
    const uint8_t *input,
    int input_height,
    int input_width,
    uint8_t *output,
    bfsrcnn_hw_stage_callback_t callback,
    void *callback_user
);

#endif
