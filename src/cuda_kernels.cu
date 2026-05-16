#include "axon/kernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>

static cublasHandle_t cublas_handle = nullptr;

void init_cublas() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
}

namespace axon::kernels::gpu {
    inline void get_launch_config(size_t n, int& blocks, int& threads) {
        threads = 256;
        blocks = (n + threads - 1) / threads;
    }

    __global__ void add_kernel(size_t n, const float* a, const float* b, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = a[idx] + b[idx];
    }

    void add_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        int b_dim, t_dim;
        get_launch_config(n, b_dim, t_dim);
        add_kernel<<<b_dim, t_dim>>>(n, a, b, out);
    }

    __global__ void sub_kernel(size_t n, const float* a, const float* b, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = a[idx] - b[idx];
    }

    void sub_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        int b_dim, t_dim;
        get_launch_config(n, b_dim, t_dim);
        sub_kernel<<<b_dim, t_dim>>>(n, a, b, out);
    }

    __global__ void mul_kernel(size_t n, const float* a, const float* b, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = a[idx] * b[idx];
    }

    void mul_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        int b_dim, t_dim;
        get_launch_config(n, b_dim, t_dim);
        mul_kernel<<<b_dim, t_dim>>>(n, a, b, out);
    }

    __global__ void div_kernel(size_t n, const float* a, const float* b, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = a[idx] / b[idx];
    }

    void div_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        int b_dim, t_dim;
        get_launch_config(n, b_dim, t_dim);
        div_kernel<<<b_dim, t_dim>>>(n, a, b, out);
    }

    __global__ void fill_kernel(size_t n, float val, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = val;
    }

    void fill_f32(size_t n, float value, float* AXON_RESTRICT out) noexcept {
        int b_dim, t_dim;
        get_launch_config(n, b_dim, t_dim);
        fill_kernel<<<b_dim, t_dim>>>(n, value, out);
    }


    /*
        MATRIX MULTIPLICATION
        Axon -> ROW MAJOR => C = A * B
        cuBLAS -> Col major => C.T = B.T * A.T
    */

    void matmul_f32(
        size_t M, size_t N, size_t K,
        const float* AXON_RESTRICT a,
        const float* AXON_RESTRICT b,
        float* AXON_RESTRICT out
    ) noexcept {

        init_cublas();
        float alpha = 1.0f;
        float beta = 0.0f;

        cublasStatus_t status = cublasSgemm(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            b, N,
            a, K, 
            &beta,
            out, N
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS Error in matmul" << std::endl;
        }

        cudaDeviceSynchronize();
    }

    __global__ void relu_kernel(size_t n, const float* inp, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = inp[idx] > 0.0f ? inp[idx] : 0.0f;
    }

    void relu_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) noexcept {
        int b, t; 
        get_launch_config(n, b, t);
        relu_kernel<<<b, t>>>(n, input, out);
    }

    __global__ void relu_back_kernel(size_t n, const float* inp, const float* grad_out, float* grad_inp) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) grad_inp[idx] = (inp[idx] > 0.0f) ? grad_out[idx] : 0.0f;
    }

    void relu_backward_f32(size_t n, const float* AXON_RESTRICT input, const float* AXON_RESTRICT grad_out, float* AXON_RESTRICT grad_input) noexcept {
        int b, t; 
        get_launch_config(n, b, t);
        relu_back_kernel<<<b, t>>>(n, input, grad_out, grad_input);
    }

    __global__ void gelu_kernel(size_t n, const float* inp, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x = inp[idx];
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            float t = tanh(inner);
            out[idx] = 0.5f * x * (1.0f + t);
        }
    }

    void gelu_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        int b, t; 
        get_launch_config(n, b, t);
        gelu_kernel<<<b, t>>>(n, input, output);
    }
    
    __global__ void neg_kernel(size_t n, const float* inp, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = -inp[idx];
    }

    void neg_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        int b, t; 
        get_launch_config(n, b, t);
        neg_kernel<<<b, t>>>(n, input, output);
    }

    __global__ void exp_kernel(size_t n, const float* inp, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) out[idx] = expf(inp[idx]);
    }

    void exp_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        int b, t; 
        get_launch_config(n, b, t);
        exp_kernel<<<b, t>>>(n, input, output);
    }

    // ! PLACEHOLDERS

    __global__ void embed_fwd_kernel(
        size_t total_elements, 
        size_t dim, size_t vocab_size, 
        const float* AXON_RESTRICT weight, 
        const float* AXON_RESTRICT indices, 
        float* AXON_RESTRICT out) {

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < total_elements) {
            // token in the sequence
            int token_seq_idx = idx / dim;
            // dimension of embedding vector
            int embed_idx = idx % dim;

            int vocab_idx = (int)indices[token_seq_idx];
            if (vocab_size >= 0 && vocab_idx < vocab_size) {
                out[idx] = weight[vocab_idx * dim + embed_idx];
            } else {
                out[idx] = 0.0f;
            }
        }
    }

    void embedding_forward_f32(
        size_t vocab_size, size_t dim, size_t num_indices, 
        const float* AXON_RESTRICT weight, 
        const float* AXON_RESTRICT indices, 
        float* AXON_RESTRICT out) noexcept {

        size_t total = num_indices * dim;
        int b, t;
        get_launch_config(total, b, t);

        embed_fwd_kernel<<<b, t>>>(total, dim, vocab_size, weight, indices, out);
    }
    

    __global__ void embed_back_kernel(
        size_t num_indices, size_t dim, size_t vocab_size, 
        const float* AXON_RESTRICT grad_out, 
        const float* AXON_RESTRICT indices, 
        float* AXON_RESTRICT grad_weight) {
        
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_indices * dim) {
            int i = idx / dim;
            int d = idx % dim;

            int vocab_idx = (int)indices[i];

            if (vocab_size >= 0 && vocab_idx < vocab_size) {
                atomicAdd(&grad_weight[vocab_idx * dim + d], grad_out[idx]);
            }
        }
    }

    void embedding_backward_f32(
        size_t vocab_size, size_t dim, size_t num_indices, 
        const float* AXON_RESTRICT grad_output, 
        const float* AXON_RESTRICT indices, 
        float* AXON_RESTRICT grad_weight) noexcept {

        size_t total = num_indices * dim;
        int b, t;
        get_launch_config(total, b, t);

        embed_back_kernel<<<b, t>>> (num_indices, dim, vocab_size, grad_output, indices, grad_weight);
    }

    
    __global__ void softmax_kernel(size_t rows, size_t cols, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) {
        int row = blockIdx.x;
        if (row >= rows) return;

        float max_val = -__FLT_MAX__;

        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float val = input[row * cols + c];
            max_val = max(max_val, val);
        }

        __shared__ float sdata[1024];
        sdata[threadIdx.x] = max_val;
        __syncthreads();


        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
            }
            __syncthreads();
        }

        max_val = sdata[0];

        float sum = 0.0f;

        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            sum += expf(input[row * cols + c] - max_val);
        }

        sdata[threadIdx.x] = sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }

            __syncthreads();
        }

        sum = sdata[0];

        float inv = 1.0f / sum;

        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            out[row * cols + c] = expf(input[row * cols + c] - max_val) * inv;
        }
    }
    
    void softmax_f32(size_t rows, size_t cols, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) noexcept {
        int threads = 256;
        if (cols < 256) threads = 128;
        if (cols >= 1024) threads = 1024;

        softmax_kernel<<<rows, threads>>>(rows, cols, input, out);
    }


    __global__ void layernorm_fwd_kernel(
        size_t rows, size_t cols,
        const float* AXON_RESTRICT input,
        const float* AXON_RESTRICT gamma,
        const float* AXON_RESTRICT beta,
        float* AXON_RESTRICT out,
        float eps
    ) {

        int row = blockIdx.x;
        if (row >= rows) return;

        float sum = 0.0f;

        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            sum += input[row * cols + c];
        }

        __shared__ float sdata[1024];
        sdata[threadIdx.x] = sum;

        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }

            __syncthreads();
        }

        float mean = sdata[0] / cols;

        float sq_diff_sum = 0.0f;

        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float diff = input[row * cols + c] - mean;
            sq_diff_sum += diff * diff;
        }
        
        sdata[threadIdx.x] = sq_diff_sum;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }

            __syncthreads();
        }

        float var = sdata[0] / cols;
        float inv_std = rsqrtf(var + eps);

        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            float val = (input[row * cols + c] - mean) * inv_std;
            out[row * cols + c] = val * gamma[c] + beta[c];
        }
    }

    void layernorm_forward_f32(
        size_t rows, size_t cols, 
        const float* AXON_RESTRICT input, 
        const float* AXON_RESTRICT gamma, 
        const float* AXON_RESTRICT beta, 
        float* AXON_RESTRICT out, float eps) noexcept {

        
        int threads = (cols > 1024) ? 1024 : 256;
        layernorm_fwd_kernel<<<rows, threads>>>(rows, cols, input, gamma, beta, out, eps);
    }
    


    void softmax_backward_f32(size_t rows, size_t cols, const float* AXON_RESTRICT grad_output, const float* AXON_RESTRICT output, float* AXON_RESTRICT grad_input) noexcept {}
    void log_softmax_f32(size_t rows, size_t cols, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) noexcept {}
    void log_softmax_backward_f32(size_t rows, size_t cols, const float* AXON_RESTRICT grad_output, const float* AXON_RESTRICT output, float* AXON_RESTRICT grad_input) noexcept {}
    void layernorm_backward_f32(size_t rows, size_t cols, const float* AXON_RESTRICT grad_out, const float* AXON_RESTRICT input, const float* AXON_RESTRICT gamma, float eps, float* AXON_RESTRICT grad_input, float* AXON_RESTRICT grad_gamma, float* AXON_RESTRICT grad_beta) noexcept {}
    void sum_f32(size_t n, const float* AXON_RESTRICT inp, float* AXON_RESTRICT out) noexcept {}
    void sum_dim_f32(size_t outer, size_t dim, size_t inner, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {}
    void sqrt_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {}
    void gelu_backward_f32(size_t n, const float* AXON_RESTRICT input, const float* AXON_RESTRICT grad_out, float* AXON_RESTRICT grad_input) noexcept {}

}




