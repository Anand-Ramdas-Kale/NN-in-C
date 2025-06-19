#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <limits.h>
#include <stdint.h>

#define MB (1024 * 1024)

struct NN {
    uint32_t *neurons;
    void **weights;
    void **biases; 
    void *mem_limit;
    uint32_t layers;
    uint32_t padding;
};

typedef struct NN NN;

__global__ void sigmoid_kernel(float *V, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        V[index] = 1.0 / (1 + expf(-V[index]));
    }
}

NN *nn_create(const uint32_t *arch, const uint32_t size) {
    // first layer is input layer, it has no biases
    // it is not counted in layers of structure
    uint32_t numWeights = 0;
    uint32_t numBiases = 0;
    for (uint32_t i = 1; i < size; ++i) {
        numWeights += arch[i - 1] * arch[i];
        numBiases += arch[i];
    }
    NN *new_nn = (NN *)malloc(sizeof(NN) + size * sizeof(uint32_t) + ((size << 1) - 2) * sizeof(void *));
    new_nn->weights = (void **)(new_nn + 1);
    new_nn->biases  = (void **)(new_nn->weights + size - 1);
    new_nn->neurons = (uint32_t *)(new_nn->biases + size - 1);
    new_nn->layers  = size - 1;
    memcpy(new_nn->neurons, arch, size * sizeof(uint32_t));

    float *pool;
    cudaMalloc((void **)&pool, sizeof(float) * (numWeights + numBiases));
    new_nn->weights[0] = pool;
    new_nn->biases[0]  = pool + numWeights;
    void **weights     = new_nn->weights;
    void **biases      = new_nn->biases;
    for (int i = 1; i < size - 1; ++i) {
        weights[i] = (float *)weights[i - 1] + arch[i] * arch[i - 1];
        biases[i]  = (float *)biases[i - 1] + arch[i];
    }
    new_nn->mem_limit = pool + numWeights + numBiases;
    return new_nn;
}

/*
 * under construction
__global__ void random(float *A) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    float random_float = 
    A[index]
}
*/

void nn_random(NN *nn, float low, float high) {
    srand(time(NULL));
    uint64_t total = ((uint64_t)nn->mem_limit - (uint64_t)nn->weights[0]);
    float *A = (float *)malloc(total);
    total >>= 2;
    for (uint64_t i = 0; i < total; ++i) {
        A[i] = (rand() * 1.0f / RAND_MAX) * (high - low) + low;
    }
    cudaMemcpy(*(nn->weights), A, total << 2, cudaMemcpyHostToDevice);
    free(A);
}

void nn_print(NN *nn) {
    // completely unoptimized
    uint32_t layers = nn->layers;
    uint32_t *neurons = nn->neurons;
    uint64_t total = ((uint64_t)(nn->mem_limit) - (uint64_t)(nn->weights[0]));
    float *A = (float *)malloc(total);
    float *W = A;
    float *B = A + (((uint64_t)(nn->biases) - (uint64_t)(nn->weights)) >> 2);
    cudaMemcpy(A, *(nn->weights), total, cudaMemcpyDeviceToHost);

    printf("\n");
    for (uint32_t layer = 0; layer < layers; ++layer) {
        uint32_t cols = neurons[layer];
        uint32_t rows = neurons[layer + 1];
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                printf("%3.2f ", W[r + c * rows]);
            }
            printf("\t\t%3.2f\n", B[r]);
        }
        W += rows * cols;
        B += rows;
        printf("\n");
    }
    free(A);
}

void nn_destroy(NN *nn) {
    if (nn) {
        cudaFree(nn->weights);
        free(nn);
    }
}

int main() {

// --------------------------------------------------------------------------------------------------------//
    // cublas inintiation
    cublasHandle_t handle;
    if (cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS) printf("Handle Created\n");
    else {
        printf("Error Creating Handle\n");
        abort();
    }
    cudaStream_t streamId;
    if (cudaStreamCreate(&streamId) == cudaSuccess) printf("Stream Created\n");
    else {
        printf("Error Creating Stream\n");
        abort();
    }
    if (cublasSetStream(handle, streamId) == CUBLAS_STATUS_SUCCESS) printf("Stream Set OK\n");
    else {
        printf("Error Setting Stream\n");
        abort();
    }
    void* workspace_ptr;
    cudaMalloc(&workspace_ptr, 8 * MB);
    if (cublasSetWorkspace(handle, workspace_ptr, 8 * MB) == CUBLAS_STATUS_SUCCESS) printf("Workspace Given\n");
    else {
        printf("Error Giving Workspace\n");
        abort();
    }

// --------------------------------------------------------------------------------------------------------//
    // rough
    /*
    void *D_Y = NULL;
    cudaMalloc(&D_Y, sizeof(float[10]));
    float *H_Y = (float *)malloc(sizeof(float[10]));
    for (int i = 0; i < 10; ++i) {
        H_Y[i] = i;
    }
    cudaMemcpy(D_Y, H_Y, sizeof(float[10]), cudaMemcpyHostToDevice);
    sigmoid_kernel<<<1, 10>>>((float *)D_Y, 10);
    cudaMemcpy(H_Y, D_Y, sizeof(float[10]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i) {
        printf("%f ", H_Y[i]);
    }
    */

// --------------------------------------------------------------------------------------------------------//
    // NN functions rough
    uint32_t a[] = {5,3,4,4,10};
    NN *nn = nn_create(a, sizeof(a) / sizeof(a[0]));
    for (int i = 0; i < nn->layers; ++i) {
        printf("%zu %zu\n", nn->weights[i], nn->biases[i]);
    }
    nn_random(nn, 0, 1);
    nn_print(nn);
    nn_destroy(nn);
    cublasDestroy(handle);
    return 0;
}
