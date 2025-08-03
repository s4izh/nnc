#ifndef TENSOR_H
#define TENSOR_H

#include "types.h"

#ifndef NNC_MALLOC
#include <stdlib.h>
#define NNC_MALLOC malloc
#endif

#ifndef NNC_FREE
#include <stdlib.h>
#define NNC_FREE free
#endif

#ifndef NNC_ASSERT
#include <assert.h>
#define NNC_ASSERT assert
#endif

#define TENSOR_MAX_DIM 4

typedef struct {
    u8 ndim;
    u8 shape[TENSOR_MAX_DIM];
    u8 stride[TENSOR_MAX_DIM];
    u32 size;
    float* data;
    bool view;
} tensor_t;

typedef tensor_t row_t;
typedef tensor_t mat_t;

void tensor_alloc_view(tensor_t* tensor, u8 ndim, const u8* shape);
void tensor_alloc(tensor_t* tensor, u8 ndim, const u8* shape);
void tensor_free(tensor_t* tensor);
void tensor_rand(tensor_t* tensor, float low, float high);
void tensor_fill(tensor_t* tensor, float value);

#endif // TENSOR_H

#ifdef TENSOR_H_IMPLEMENTATION

// MAT utils
#define MAT_AT(tensor, i, j) \
    ((tensor)->data[(i) * (tensor)->stride[0] + (j) * (tensor)->stride[1]])

#define MAT_ALLOC(_tensor, _rows, _cols)                                       \
    do {                                                                       \
        u8 _shape[2] = {_rows, _cols};                                         \
        tensor_alloc(_tensor, 2, _shape);                                      \
    } while (0)

#define MAT_PRINT(mat) tensor_print(mat, #mat, false)

#define MAT_RAND(_tensor, _low, _high) tensor_rand(_tensor, _low, _high)
#define MAT_FILL(_tensor, _value) tensor_fill(_tensor, _value)

#define MAT_VIEW(_tensor, _data_ptr, _rows, _cols, _stride_rows, _stride_cols) \
    do {                                                                       \
        const u8 _shape[2] = {_rows, _cols};                                   \
        tensor_alloc_view(_tensor, 2, _shape);                                 \
        (_tensor)->data = (_data_ptr); /* Attach the external data */          \
        (_tensor)->stride[0] = (_stride_rows);                                 \
        (_tensor)->stride[1] = (_stride_cols);                                 \
    } while (0)

#define MAT_DOT(_dst, _src1, _src2) tensor_2d_dot_product(_dst, _src1, _src2)
#define MAT_SUM(_dst, _mat) tensor_2d_sum(_dst, _mat)
#define MAT_COPY(_dst, _src) tensor_copy(_dst, _src)
#define MAT_ACT(_dst, _func) tensor_activate(_dst, _func)
#define MAT_ROWS(mat) (mat)->shape[0]
#define MAT_COLS(mat) (mat)->shape[1]

// ROW utils
#define ROW_VIEW(dst, src, row) tensor_2d_to_1d_view(dst, src, row)
#define ROW_COPY(dst, src) tensor_copy(dst, src)
#define ROW_AT(row, j) ((row).data[(j) * (row).stride[0]])

float randf(void)
{
    return (float) rand() / (float) RAND_MAX;
}

float randf_ranged(float low, float high)
{
    return randf() * (high - low) + low;
}

void tensor_alloc_view(tensor_t* tensor, u8 ndim, const u8* shape)
{
    assert(tensor != NULL);
    assert(ndim != TENSOR_MAX_DIM);
    assert(shape != NULL);

    tensor->ndim = ndim;

    size_t size = 1;
    for (int i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        size *= shape[i];
    }

    size_t stride_val = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->stride[i] = stride_val;
        stride_val *= tensor->shape[i];
    }

    tensor->data = NULL;
    tensor->size = size;
    tensor->view = true;
}

void tensor_alloc(tensor_t* tensor, u8 ndim, const u8* shape)
{
    tensor_alloc_view(tensor, ndim, shape);

    tensor->data = NNC_MALLOC(tensor->size * sizeof(float));
    assert(tensor->data != NULL);
    tensor->view = false;
}

void tensor_free(tensor_t* tensor)
{
    if (tensor == NULL || tensor->data == NULL)
        return;

    if (!tensor->view)
        NNC_FREE(tensor->data);

    tensor->data = NULL;
}

void tensor_rand(tensor_t* tensor, float low, float high)
{
    for (size_t i = 0; i < tensor->size; ++i) {
        tensor->data[i] = randf_ranged(low, high);
    }
}

void tensor_fill(tensor_t* tensor, float value)
{
    for (size_t i = 0; i < tensor->size; ++i) {
        tensor->data[i] = value;
    }
}

void tensor_copy(tensor_t* dst, const tensor_t* src)
{
    NNC_ASSERT(dst != NULL && src != NULL);
    NNC_ASSERT(dst->data != NULL && src->data != NULL);
    NNC_ASSERT(dst->size == src->size);

    if (src->size == 0) return;

    if (dst->ndim == 2 && src->ndim == 1 && dst->shape[0] == 1) {
        for (u32 i = 0; i < src->shape[0]; ++i) {
            dst->data[i * dst->stride[1]] = src->data[i * src->stride[0]];
        }
    }
    else if (dst->ndim == 1 && src->ndim == 2 && src->shape[0] == 1) {
         for (u32 i = 0; i < dst->shape[0]; ++i) {
            dst->data[i * dst->stride[0]] = src->data[i * src->stride[1]];
        }
    }
    else if (dst->ndim == 2 && dst->ndim == src->ndim) {
        NNC_ASSERT(dst->shape[0] == src->shape[0]);
        NNC_ASSERT(dst->shape[1] == src->shape[1]);
        for (u32 i = 0; i < dst->shape[0]; ++i) {
            for (u32 j = 0; j < dst->shape[1]; ++j) {
                float* src_ptr = &src->data[i * src->stride[0] + j * src->stride[1]];
                float* dst_ptr = &dst->data[i * dst->stride[0] + j * dst->stride[1]];
                *dst_ptr = *src_ptr;
            }
        }
    }
    else {
        NNC_ASSERT(false && "tensor_copy: Unhandled dimension combination");
    }
}

void tensor_2d_to_1d_view(tensor_t* dst, const tensor_t* src, u32 row)
{
    NNC_ASSERT(dst != NULL && src != NULL);
    NNC_ASSERT(src->data != NULL);
    NNC_ASSERT(src->ndim == 2);
    NNC_ASSERT(row < MAT_ROWS(src));

    dst->ndim = 1;
    dst->view = true;

    dst->size = MAT_COLS(src);
    
    dst->shape[0] = MAT_COLS(src);
    dst->stride[0] = MAT_COLS(src);
    dst->data = &src->data[row * src->stride[0]];
}

void tensor_2d_dot_product(tensor_t* dst, const tensor_t* src1, const tensor_t* src2)
{
    NNC_ASSERT(dst != NULL && src1 != NULL && src2 != NULL);
    NNC_ASSERT(MAT_COLS(src1) == MAT_ROWS(src2));
    NNC_ASSERT(MAT_ROWS(dst) == MAT_ROWS(src1));
    NNC_ASSERT(MAT_COLS(dst) == MAT_COLS(src2));
    
    for (u32 i = 0; i < MAT_ROWS(src1); ++i) {
        for (u32 j = 0; j < MAT_COLS(src2); ++j) {
            float sum = 0.0f;
            for (u32 k = 0; k < MAT_COLS(src1); ++k) {
                sum += MAT_AT(src1, i, k) * MAT_AT(src2, k, j);
            }
            MAT_AT(dst, i, j) = sum;
        }
    }
}

void tensor_2d_sum(tensor_t* dst, tensor_t* a)
{
    NNC_ASSERT(dst->shape[0] == a->shape[0]);
    NNC_ASSERT(dst->shape[1] == a->shape[1]);
    for (size_t i = 0; i < dst->shape[0]; ++i) {
        for (size_t j = 0; j < dst->shape[1]; ++j) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

void tensor_activate(tensor_t* dst, float (*activate)(float))
{
    NNC_ASSERT(dst != NULL);
    NNC_ASSERT(dst->data != NULL);

    for (size_t i = 0; i < dst->shape[0]; ++i) {
        for (size_t j = 0; j < dst->shape[1]; ++j) {
            MAT_AT(dst, i, j) = (activate)(MAT_AT(dst, i, j));
        }
    }
}

void tensor_print(const tensor_t* tensor, const char* name, bool detailed)
{
    assert(tensor != NULL);
    assert(name != NULL);
    
    printf("%s = [\n", name);
    for(int i = 0; i < tensor->shape[0]; i++) {
        printf("    ");
        for (int j = 0; j < tensor->shape[1]; j++) {
            printf("%5.1f ", tensor->data[i * tensor->stride[0] + j * tensor->stride[1]]);
            // printf("%f", tensor->data[i * tensor->stride[0] + j * tensor->stride[1]]);
            if (j != (tensor->shape[1] - 1))
                printf("    ");
        }
        printf("\n");
    }
    printf("]\n");
}

#endif // TENSOR_H_IMPLEMENTATION
