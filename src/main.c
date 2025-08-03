#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define TENSOR_H_IMPLEMENTATION
#include "tensor.h"

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x)); 
}

typedef struct {
    tensor_t as;
    tensor_t ws;
    tensor_t bs;
    float (*activate)(float z);
} layer_t;

typedef struct {
    size_t *arch;
    size_t arch_count;
#if 0
    tensor_t* ws; // arch_count - 1
    tensor_t* bs; // arch_count - 1
    tensor_t* as; // arch_count 
#else
    layer_t* layers; // arch_count
#endif
} nn_t;

#define NN_INPUT(nn) ((nn)->layers[0].as)
#define NN_OUTPUT(nn) ((nn)->layers[(nn)->arch_count-1].as)

float or_train[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 1,
};

float and_train[] = {
    0, 0, 0,
    0, 1, 0,
    1, 0, 0,
    1, 1, 1,
};

#define TRAIN_COUNT 4
#define TRAIN_FEATURES 2
#define TRAIN_LABEL 1

float* train = or_train;

void nn_alloc(nn_t* nn, size_t arch[], size_t arch_count)
{
    nn->arch = arch;
    nn->arch_count = arch_count;

#if 0
    nn->ws = malloc(sizeof(*nn->ws) * (nn->arch_count - 1));
    nn->bs = malloc(sizeof(*nn->bs) * (nn->arch_count - 1));
    nn->as = malloc(sizeof(*nn->as) * nn->arch_count);

    MAT_ALLOC(&nn->as[0], arch[0], 1);
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_ALLOC(&nn->ws[i-1], MAT_COLS(&nn->as[0]), arch[i]);
        MAT_ALLOC(&nn->bs[i-1], arch[i], 1);
        MAT_ALLOC(&nn->as[i], arch[i], 1);
    }
#else
    nn->layers = malloc(sizeof(*nn->layers) * (nn->arch_count));
    MAT_ALLOC(&nn->layers[0].as, 1, arch[0]);
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_ALLOC(&nn->layers[i].ws, MAT_COLS(&nn->layers[i-1].as), arch[i]);
        MAT_ALLOC(&nn->layers[i].bs, 1, arch[i]);
        MAT_ALLOC(&nn->layers[i].as, 1, arch[i]);
        nn->layers[i].activate = &sigmoidf;
    };
#endif
}

void nn_print(nn_t* nn)
{
    char buf[64];
    sprintf(buf, "as[%d]", 0);
    tensor_print(&nn->layers[0].as, buf, false);
    for (size_t i = 1; i < nn->arch_count; ++i) {
        sprintf(buf, "ws[%ld]", i);
        tensor_print(&nn->layers[i].ws, buf, false);
        sprintf(buf, "bs[%ld]", i);
        tensor_print(&nn->layers[i].bs, buf, false);
    }
    sprintf(buf, "as[%ld]", nn->arch_count - 1);
    tensor_print(&nn->layers[nn->arch_count - 1].as, buf, false);
}

void nn_rand(nn_t* nn, float low, float high)
{
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_RAND(&nn->layers[i].bs, low, high);
        MAT_RAND(&nn->layers[i].ws, low, high);
    }
}

void nn_forward(nn_t* nn)
{
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_DOT(&nn->layers[i].as, &nn->layers[i-1].as, &nn->layers[i].ws);
        MAT_SUM(&nn->layers[i].as, &nn->layers[i].bs);
        MAT_ACT(&nn->layers[i].as, nn->layers[i].activate);
    }
}

void nn_cost(nn_t* nn, tensor_t* target)
{
    tensor_t* output = &nn->layers[nn->arch_count - 1].as;
    NNC_ASSERT(output->shape[0] == target->shape[0]);
    NNC_ASSERT(output->shape[1] == target->shape[1]);

    float cost = 0.0f;
    for (size_t i = 0; i < output->shape[0]; ++i) {
        for (size_t j = 0; j < output->shape[1]; ++j) {
            float diff = MAT_AT(output, i, j) - MAT_AT(target, i, j);
            cost += diff * diff;
        }
    }
    cost /= MAT_ROWS(output);
    printf("Cost: %f\n", cost);
}

void nn_finite_diff(nn_t* nn)
{

}

#define ARRAY_LEN(_arr) sizeof(_arr)/sizeof(_arr[0])

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    printf("Hello, World!\n");

    tensor_t mat;
    MAT_ALLOC(&mat, 3, 4);

    u8 stride = 3;

    tensor_t input;
    MAT_VIEW(&input, train, TRAIN_COUNT, TRAIN_FEATURES, stride, 1);

    tensor_t output;
    MAT_VIEW(&output, train + 2, TRAIN_COUNT, TRAIN_LABEL, stride, 1);

    MAT_PRINT(&input);
    MAT_PRINT(&output);

    srand(0);
    nn_t nn_test;

    size_t arch[] = {2, 2, 1};

    nn_alloc(&nn_test, arch, ARRAY_LEN(arch));
    nn_rand(&nn_test, 0, 1);

    for (size_t i = 0; i < TRAIN_COUNT; ++i) {
        row_t* nn_input = &NN_INPUT(&nn_test);

        tensor_t in_row;
        ROW_VIEW(&in_row, &input, i);
        ROW_COPY(nn_input, &in_row);

        nn_forward(&nn_test);
        nn_print(&nn_test);

        tensor_free(&in_row);
        break;
    }
    // nn_forward(&nn_test);
}
