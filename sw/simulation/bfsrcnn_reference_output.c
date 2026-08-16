/* Emit BFSRCNN final golden output. */
#include "bfsrcnn_reference.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

static int write_exact(const char *path, const void *data, size_t size)
{
    FILE *file = fopen(path, "wb");
    int status = 0;
    if (file == NULL) {
        return -1;
    }
    if (fwrite(data, size, 1u, file) != 1u) {
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
    int status;

    if (argc != 5) {
        fprintf(
            stderr,
            "usage: %s HEIGHT WIDTH INPUT OUTPUT\n",
            argv[0]);
        return 2;
    }
    height = atoi(argv[1]);
    width = atoi(argv[2]);
    if (height <= 0 || width <= 0) {
        return 3;
    }
    input_size = (size_t)height * (size_t)width;
    output_size = input_size * 16u;
    input = malloc(input_size);
    output = malloc(output_size);
    if (input == NULL || output == NULL
        || read_exact(argv[3], input, input_size) != 0) {
        free(output);
        free(input);
        return 4;
    }
    status = bfsrcnn_hw_infer(input, height, width, output);
    if (status == 0) {
        status = write_exact(argv[4], output, output_size);
    }
    free(output);
    free(input);
    return status == 0 ? 0 : 5;
}
