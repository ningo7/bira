/* Emit BFSRCNN intermediate golden outputs. */
#include "bfsrcnn_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    FILE *file;
    int error;
} stage_writer_t;

static void write_stage(
    bfsrcnn_hw_stage_t stage,
    const int32_t *data,
    int height,
    int width,
    int channels,
    void *user)
{
    stage_writer_t *writer = (stage_writer_t *)user;
    size_t count =
        (size_t)height * (size_t)width * (size_t)channels;
    size_t index;
    (void)stage;
    for (index = 0u; index < count; ++index) {
        uint8_t value = (uint8_t)data[index];
        if (fwrite(&value, sizeof(value), 1u, writer->file) != 1u) {
            writer->error = 1;
            return;
        }
    }
}

static int read_exact(const char *path, void *data, size_t size)
{
    FILE *file = fopen(path, "rb");
    int status = 0;
    if (file == NULL) {
        return -1;
    }
    if (fread(data, size, 1u, file) != 1u || fgetc(file) != EOF) {
        status = -1;
    }
    if (fclose(file) != 0) {
        status = -1;
    }
    return status;
}

int main(int argc, char **argv)
{
    int height;
    int width;
    size_t input_size;
    size_t output_size;
    uint8_t *input;
    uint8_t *output;
    stage_writer_t writer;
    int status;

    if (argc != 5) {
        fprintf(
            stderr,
            "usage: %s HEIGHT WIDTH INPUT STAGES\n",
            argv[0]);
        return 2;
    }
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    input_size = (size_t)height * (size_t)width;
    output_size = input_size * 16u;
    input = malloc(input_size);
    output = malloc(output_size);
    writer.file = fopen(argv[4], "wb");
    writer.error = 0;
    if (height <= 0 || width <= 0 || input == NULL || output == NULL
        || writer.file == NULL
        || read_exact(argv[3], input, input_size) != 0) {
        if (writer.file != NULL) {
            fclose(writer.file);
        }
        free(output);
        free(input);
        return 3;
    }
    status = bfsrcnn_hw_infer_stages(
        input,
        height,
        width,
        output,
        write_stage,
        &writer);
    if (fclose(writer.file) != 0) {
        writer.error = 1;
    }
    free(output);
    free(input);
    return status == 0 && !writer.error ? 0 : 4;
}
