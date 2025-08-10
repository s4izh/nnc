#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

#define TENSOR_H_IMPLEMENTATION
#include "tensor.h"

#include "hrtimer.h"

float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x)); 
}

float sigmoidf_derivative(float x_sigmoid)
{
    return x_sigmoid * (1 - x_sigmoid); 
}

typedef struct {
    tensor_t as;
    tensor_t ws;
    tensor_t bs;
    float (*act)(float z);
    float (*dact)(float z);
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

float xor_train[] = {
    0, 0, 0,
    0, 1, 1,
    1, 0, 1,
    1, 1, 0,
};

#define TRAIN_COUNT 4
#define TRAIN_FEATURES 2
#define TRAIN_LABEL 1

float* train = xor_train;

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
    nn->layers = NNC_MALLOC(sizeof(*nn->layers) * (nn->arch_count));
    MAT_ALLOC(&nn->layers[0].as, 1, arch[0]);
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_ALLOC(&nn->layers[i].ws, MAT_COLS(&nn->layers[i-1].as), arch[i]);
        MAT_ALLOC(&nn->layers[i].bs, 1, arch[i]);
        MAT_ALLOC(&nn->layers[i].as, 1, arch[i]);
        nn->layers[i].act  = &sigmoidf;
        nn->layers[i].dact = &sigmoidf_derivative;
    };
#endif
}

void nn_free(nn_t* nn)
{
    MAT_FREE(&nn->layers[0].as);
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_FREE(&nn->layers[i].ws);
        MAT_FREE(&nn->layers[i].bs);
        MAT_FREE(&nn->layers[i].as);
        nn->layers[i].act = NULL;
    };
    NNC_FREE(nn->layers);
}

void nn_print(nn_t* nn)
{
    char buf[64];
    // sprintf(buf, "as[%d]", 0);
    // tensor_print(&nn->layers[0].as, buf, false);
    for (size_t i = 1; i < nn->arch_count; ++i) {
        sprintf(buf, "ws[%ld]", i);
        tensor_print(&nn->layers[i].ws, buf, false);
        sprintf(buf, "bs[%ld]", i);
        tensor_print(&nn->layers[i].bs, buf, false);
    }
    // sprintf(buf, "as[%ld]", nn->arch_count - 1);
    // tensor_print(&nn->layers[nn->arch_count - 1].as, buf, false);
}

void nn_rand(nn_t* nn, float low, float high)
{
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_RAND(&nn->layers[i].bs, low, high);
        MAT_RAND(&nn->layers[i].ws, low, high);
    }
}

void nn_fill(nn_t* nn, float value)
{
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_FILL(&nn->layers[i].bs, value);
        MAT_FILL(&nn->layers[i].ws, value);
        MAT_FILL(&nn->layers[i].as, value);
    }
}

void nn_forward(nn_t* nn)
{
    for (size_t i = 1; i < nn->arch_count; ++i) {
        MAT_DOT(&nn->layers[i].as, &nn->layers[i-1].as, &nn->layers[i].ws);
        MAT_SUM(&nn->layers[i].as, &nn->layers[i].bs);
        MAT_ACT(&nn->layers[i].as, nn->layers[i].act);
    }
}

float nn_cost(nn_t* nn, tensor_t* target)
{
    tensor_t* input_mat  = &NN_INPUT(nn);
    tensor_t* output_mat = &NN_OUTPUT(nn);
    assert((MAT_COLS(input_mat) + MAT_COLS(output_mat)) == MAT_COLS(target));

    float cost = 0.0f;

    size_t samples = MAT_ROWS(target);

    for (size_t i = 0; i < samples; ++i) {
        row_t row, x, y;
        tensor_2d_to_1d_row_view(&row, target, i);
        tensor_1d_slice(&x, &row, 0, MAT_COLS(input_mat));
        tensor_1d_slice(&y, &row, MAT_COLS(input_mat), MAT_COLS(output_mat));

        ROW_COPY(input_mat, &x);
        nn_forward(nn);

        for (size_t j = 0; j < MAT_COLS(output_mat); ++j) {
            float d =  MAT_AT(output_mat, 0, j) - ROW_AT(&y, j);
            cost += d * d;
        }
    }
    return cost /= samples;
}

void nn_finite_diff(nn_t* nn, nn_t* grad, tensor_t* target, float eps)
{
    float saved;
    float c = nn_cost(nn, target);

    for (size_t i = 1; i < nn->arch_count; ++i) {
        for (size_t j = 0; j < MAT_ROWS(&nn->layers[i].ws); ++j) {
            for (size_t k = 0; k < MAT_COLS(&nn->layers[i].ws); ++k) {
                saved = MAT_AT(&nn->layers[i].ws, j, k);
                MAT_AT(&nn->layers[i].ws, j, k) += eps;
                MAT_AT(&grad->layers[i].ws, j, k) = (nn_cost(nn, target) - c)/eps;
                MAT_AT(&nn->layers[i].ws, j, k) = saved;

            }
        }
        for (size_t j = 0; j < MAT_COLS(&nn->layers[i].bs); ++j) {
            saved = MAT_AT(&nn->layers[i].bs, 0, j);
            MAT_AT(&nn->layers[i].bs, 0, j) += eps;
            MAT_AT(&grad->layers[i].bs, 0, j) = (nn_cost(nn, target) - c)/eps;
            MAT_AT(&nn->layers[i].bs, 0, j) = saved;
        }
    }
}

void nn_backprop(nn_t* nn, nn_t* grad, tensor_t* target)
{
    assert(MAT_COLS(&NN_INPUT(nn)) + MAT_COLS(&NN_OUTPUT(nn)) == MAT_COLS(target));

    size_t samples = MAT_ROWS(target);

    tensor_t* input_mat  = &NN_INPUT(nn);
    tensor_t* output_mat = &NN_OUTPUT(nn);

    nn_fill(grad, 0.0f);

    for (size_t i = 0; i < samples; ++i) {
        row_t row, x, y;
        tensor_2d_to_1d_row_view(&row, target, i);
        tensor_1d_slice(&x, &row, 0, MAT_COLS(input_mat));
        tensor_1d_slice(&y, &row, MAT_COLS(input_mat), MAT_COLS(output_mat));
        ROW_COPY(input_mat, &x);
        nn_forward(nn);

        for (size_t l = 0; l < nn->arch_count; ++l) {
            MAT_FILL(&grad->layers[l].as, 0.0f);
        }

        // compute the last layer activation gradient
        size_t last_layer = nn->arch_count - 1;
        for (size_t j = 0; j < MAT_COLS(&NN_OUTPUT(nn)); ++j) {
            float a = MAT_AT(&NN_OUTPUT(nn), 0, j);
            float expected = ROW_AT(&y, j);
            MAT_AT(&grad->layers[last_layer].as, 0, j) = 2.0f * (a - expected);
        }

        for (size_t l = nn->arch_count - 1; l > 0; --l) {
            for (size_t j = 0; j < MAT_COLS(&nn->layers[l].as); ++j) {
                float a = MAT_AT(&nn->layers[l].as, 0, j);
                float dC_da = MAT_AT(&grad->layers[l].as, 0, j);
                float da_dz = (nn->layers[l].dact)(a);

                float delta = dC_da * da_dz;
                MAT_AT(&grad->layers[l].bs, 0, j) += delta;

                // iterate over neurons in the previous layer 'l-1'
                for (size_t k = 0; k < MAT_COLS(&nn->layers[l-1].as); ++k) {
                    float prev_a = MAT_AT(&nn->layers[l-1].as, 0, k); // a_{l-1}
                    float w = MAT_AT(&nn->layers[l].ws, k, j);        // w_kj

                    // accumulate gradient for the weight (dC/dw = dC/dz * a_{l-1})
                    MAT_AT(&grad->layers[l].ws, k, j) += delta * prev_a;

                    // propagate error to the previous layer's activation gradient
                    // (dC/da_{l-1} = SUM over j of dC/dz_l * w_kj)
                    MAT_AT(&grad->layers[l-1].as, 0, k) += delta * w;
                }
            }
        }
    }
    for (size_t l = 1; l < nn->arch_count; ++l) {
        for (size_t j = 0; j < MAT_ROWS(&grad->layers[l].ws); ++j) {
            for (size_t k = 0; k < MAT_COLS(&grad->layers[l].ws); ++k) {
                MAT_AT(&grad->layers[l].ws, j, k) /= samples;
            }
        }
        for (size_t j = 0; j < MAT_COLS(&grad->layers[l].bs); ++j) {
            MAT_AT(&grad->layers[l].bs, 0, j) /= samples;
        }
    }
}

void nn_learn(nn_t* nn, nn_t* grad, float rate)
{
    for (size_t l = 1; l < nn->arch_count; ++l) {
        for (size_t j = 0; j < MAT_ROWS(&nn->layers[l].ws); ++j) {
            for (size_t k = 0; k < MAT_COLS(&nn->layers[l].ws); ++k) {
                MAT_AT(&nn->layers[l].ws, j, k) -= rate * MAT_AT(&grad->layers[l].ws, j, k);
            }
        }
        for (size_t j = 0; j < MAT_COLS(&nn->layers[l].bs); ++j) {
            MAT_AT(&nn->layers[l].bs, 0, j) -= rate * MAT_AT(&grad->layers[l].bs, 0, j);
        }
    }
}

void nn_train_finite_diff(nn_t* nn, tensor_t* target, size_t epochs, float rate, float eps, size_t batch_size)
{
    nn_t grad;
    nn_alloc(&grad, nn->arch, nn->arch_count);
    nn_fill(&grad, 0);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        nn_finite_diff(nn, &grad, target, eps);
        nn_learn(nn, &grad, rate);
    }
    nn_free(&grad);
}

void nn_train(nn_t* nn, tensor_t* target, size_t epochs, float rate, size_t batch_size)
{
    nn_t grad;
    nn_alloc(&grad, nn->arch, nn->arch_count);
    nn_fill(&grad, 0);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        nn_backprop(nn, &grad, target);
        nn_learn(nn, &grad, rate);
    }
    nn_free(&grad);
}

#define ARRAY_LEN(_arr) sizeof(_arr)/sizeof(_arr[0])

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    u8 stride = 3;

    tensor_t input;
    MAT_VIEW(&input, train, TRAIN_COUNT, TRAIN_FEATURES, stride, 1);

    tensor_t output;
    MAT_VIEW(&output, train + 2, TRAIN_COUNT, TRAIN_LABEL, stride, 1);

    tensor_t target;
    MAT_VIEW(&target, train, TRAIN_COUNT, TRAIN_FEATURES + TRAIN_LABEL, stride, 1);

    srand(0);
    nn_t nn;

    size_t arch[] = {2, 2, 1};
    nn_alloc(&nn, arch, ARRAY_LEN(arch));

    nn_print(&nn);

    float rate = 1e-1;
    float eps  = 1e-3;

    nn_fill(&nn, 0);
    nn_rand(&nn, 0, 1);

    size_t finite_epoch = 100 * 1000;
    float finite_time;

    stopwatch_t sw;
    stopwatch_start(&sw);
    nn_train_finite_diff(&nn, &target, finite_epoch, rate, eps, 1);
    stopwatch_stop(&sw);

    finite_time = stopwatch_get_elapsed_seconds(&sw, get_timer_frequency());

    printf("finite diff: cost(%f), epoch(%ld), time(%f)\n", nn_cost(&nn, &target), finite_epoch, finite_time);
    printf("-----------------\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            mat_t* input  = &NN_INPUT(&nn);
            MAT_AT(input, 0, 0) = i;
            MAT_AT(input, 0, 1) = j;
            nn_forward(&nn);
            tensor_t* output = &NN_OUTPUT(&nn);
            printf("%ld %ld = %f\n", i, j, MAT_AT(output, 0, 0));
        }
    }

    size_t backprop_epoch = 1000 * 1000;
    float backprop_time;

    nn_fill(&nn, 0);
    nn_rand(&nn, 0, 1);

    stopwatch_start(&sw);
    nn_train(&nn, &target, backprop_epoch, rate, 1);
    stopwatch_stop(&sw);

    backprop_time = stopwatch_get_elapsed_seconds(&sw, get_timer_frequency());

    printf("backprop: cost(%f), epoch(%ld), time(%f)\n", nn_cost(&nn, &target), backprop_epoch, backprop_time);
    printf("-----------------\n");
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            mat_t* input  = &NN_INPUT(&nn);
            MAT_AT(input, 0, 0) = i;
            MAT_AT(input, 0, 1) = j;
            nn_forward(&nn);
            tensor_t* output = &NN_OUTPUT(&nn);
            printf("%ld %ld = %f\n", i, j, MAT_AT(output, 0, 0));
        }
    }
}
