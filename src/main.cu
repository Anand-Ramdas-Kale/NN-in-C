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

__global__ void add_kernel(float alpha, const float *A, float *B, uint32_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        B[index] = alpha * A[index] + B[index];
    }
}
__global__ void nn_last_layer_grad_b(float *Y, float *A, uint32_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float a = A[index];
        float y = Y[index];
        A[index] = (2.f / n) * a * (1 - a) * (a - y);
    }
}

__global__ void update_a2grad_b(const float *scratch_buffer, float *A, uint32_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        float a = A[index];
        float b = scratch_buffer[index];
        A[index] = a * (1 - a) * b;
    }
}

// Leaky ReLU activation function
__global__ void leaky_relu_kernel(float *V, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // You can also use fmaxf(0.01f * V[index], V[index])
        if (V[index] < 0) {
            V[index] *= 0.01f;
        }
    }
}

// Derivative of Leaky ReLU for backpropagation.
// This replaces your update_a2grad_b_relu kernel.
__global__ void update_a2grad_b_leaky_relu(const float *incoming_grad, float *activations, uint32_t n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        // Derivative is 1 if activation was > 0, else it's 0.01.
        float derivative = (activations[index] > 0.0f) ? 1.0f : 0.01f;
        activations[index] = incoming_grad[index] * derivative;
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

void nn_forward(NN *nn, float *input, float *activations, cublasHandle_t handle) {
    uint32_t *neurons  = nn->neurons;
    uint32_t layers    = nn->layers;
    float**  weights   = (float **)(nn->weights);
    float**  biases    = (float **)(nn->biases);

    float *previous_activations;
    float *next_activations = activations;
    const float one  = 1.0f;

    const uint32_t rows  = neurons[1];
    const uint32_t cols  = neurons[0];
    previous_activations = input;
    next_activations     = activations;
    uint64_t total       = ((uint64_t)nn->mem_limit - (uint64_t)nn->biases[0]);
    cudaMemcpy(next_activations, biases[0], total, cudaMemcpyDeviceToDevice);
    // I am not checking the return status, if this fails, everything should ... same in loop
    cublasSgemv(handle, CUBLAS_OP_N,
                rows, cols,
                &one,
                weights[0], rows,
                previous_activations, 1,
                &one,
                next_activations, 1);
    leaky_relu_kernel<<<rows>>>(next_activations, rows);

    for (uint32_t layer = 1; layer < layers; ++layer) {
        uint32_t rows  = neurons[layer + 1];
        uint32_t cols  = neurons[layer];
        previous_activations = next_activations;
        next_activations     = previous_activations + cols;
        // I am not checking the return status, if this fails, everything should ...
        cublasSgemv(handle, CUBLAS_OP_N,
                    rows, cols,
                    &one,
                    weights[layer], rows,
                    previous_activations, 1,
                    &one,
                    next_activations, 1);
        leaky_relu_kernel<<<rows>>>(next_activations, rows);
    }
}

void nn_learn(NN *nn, float *activations, float *input, float *expected,
              float *scratch_buffer, float learningRate, cublasHandle_t handle)
{
    uint32_t* neurons = (uint32_t *) (nn->neurons);
    float**   weights = (float **  ) (nn->weights);
    float**   biases  = (float **  ) (nn->biases );
    uint32_t   layers = (uint32_t  ) (nn->layers );
    uint64_t   size_b = (uint64_t  ) (((uint64_t) (nn->mem_limit) - (uint64_t) (biases[0])) >> 2);
    // last layer:
    uint32_t coverd = neurons[layers];
    float* gradB    = activations + size_b - coverd;
    nn_last_layer_grad_b<<<coverd>>>(expected, gradB, coverd);

    // back propogation:
    float one  = 1.f;
    float zero = 0.f;
    float rate = -1 * learningRate;

    for (int layer = layers - 1; layer > 0; --layer) {
        uint32_t cols = neurons[layer - 1];
        uint32_t rows = neurons[layer]    ;
        // add_kernel<<< (rows + 255) / 256, 256 >>>(rate, gradB, biases[layer], rows);
        // partial C by partial a for previous layer in scratch_buffer
        cublasSgemv(handle, CUBLAS_OP_T,
                    cols, rows,
                    &one,
                    weights[layer], rows,
                    gradB, 1,
                    &zero,
                    scratch_buffer, 1);

        // update weights[layer] now
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    rows, cols, 1,
                    &rate,
                    gradB, rows,
                    gradB - cols, cols,
                    &one,
                    weights[layer], rows);

        // use scratch buffer to update previous activations to grad b
        gradB = gradB - cols;
        update_a2grad_b_leaky_relu<<<cols>>>(scratch_buffer, gradB, cols);
    }

    // update the zeroth weight matrix
    uint32_t rows = neurons[1];
    uint32_t cols = neurons[0];
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                rows, cols, 1,
                &rate,
                gradB, rows,
                input, cols,
                &one,
                weights[0], rows);
    add_kernel<<<size_b>>>(rate, gradB, biases[0], size_b);
}

void nn_print(NN *nn) {
    // completely unoptimized
    uint32_t layers = nn->layers;
    uint32_t *neurons = nn->neurons;
    uint64_t total = ((uint64_t)(nn->mem_limit) - (uint64_t)(nn->weights[0]));
    float *A = (float *)malloc(total);
    float *W = A;
    float *B = A + (((uint64_t)(nn->biases[0]) - (uint64_t)(nn->weights[0])) >> 2);
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
        cudaFree(nn->weights[0]);
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
    uint32_t a[] = {2,2,1};
    uint32_t mid = 0;
    uint64_t sz  = sizeof(a) / sizeof(a[0]);
    for (int i = 1; i < sz - 1; ++i) {
        mid += a[i];
    }
    NN *nn = nn_create(a, sz);
    nn_random(nn, -1, 1);

    float *input;
    float *expected;
    float *activations;
    float *scratch_buffer;
    
    cudaMalloc(&input, sizeof(float) * 8);
    cudaMalloc(&expected, sizeof(float) * 4);
    cudaMalloc(&activations, sizeof(float) * 4);
    cudaMalloc(&scratch_buffer, sizeof(float) * 8);

    float input_h[]    = {0,0,0,1,1,0,1,1};
    float expected_h[] = {1,0,0,1};
    float learningRate = 1e-1;
    // float exact_set[ ] = {20, -20, 20, -20, 20, 20, -10, 30, -30};
    cudaMemcpy(input, input_h, sizeof(input_h), cudaMemcpyHostToDevice);
    cudaMemcpy(expected, expected_h, sizeof(expected_h), cudaMemcpyHostToDevice);
    // cudaMemcpy(nn->weights[0], exact_set, sizeof(exact_set), cudaMemcpyHostToDevice);

    nn_print(nn);
    nn_forward(nn, input, activations, handle);
    // float result = 0;
    // cudaMemcpy(&result, activations + mid, sizeof(float), cudaMemcpyDeviceToHost);
    // current_cost += (expected_h[j] - result) * (expected_h[j] - result);
    nn_learn(nn, activations, input, expected, scratch_buffer, learningRate, handle);
    nn_print(nn);
    int epochs = 100 * 1000;
    for (int i = 0; i < epochs; ++i) {
        float current_cost = 0.f;
        for (int j = 0; j < 4; ++j) {
            nn_forward(nn, input + (j << 1), activations, handle);
            float result = 0;
            cudaMemcpy(&result, activations + mid, sizeof(float), cudaMemcpyDeviceToHost);
            current_cost += (expected_h[j] - result) * (expected_h[j] - result);
            nn_learn(nn, activations, input + (j << 1), expected + j, scratch_buffer, learningRate, handle);
        }
        if (i % 1000 == 0) {
            printf("epoch %d: cost = %f\n", i, current_cost / 4.0f);
        }
    }
    for (int j = 0; j < 4; ++j) {
        float result = 0;
        nn_forward(nn, input + (j << 1), activations, handle);
        cudaMemcpy(&result, activations + mid, sizeof(float), cudaMemcpyDeviceToHost);
        printf("%.2f | %.2f = %.2f\n", input_h[j << 1], input_h[(j << 1) + 1], result);
    }

    nn_print(nn);
    nn_destroy(nn);
    cublasDestroy(handle);
    return 0;
}
