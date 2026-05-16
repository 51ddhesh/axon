#include "axon/kernels.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>
#include <iostream>

static cublasHandle_t cublas_handle = nullptr;

void init_cublas() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
}

namespace axon::kernels::gpu {

    /*******************************************
     *                                         *
     *             WARPING UTILS               *
     *                                         *
     *******************************************/


    constexpr int WARP_SIZE = 32;

    inline void get_launch_config(size_t n, int& blocks, int& threads) {
        threads = 256;
        blocks = (n + threads - 1) / threads;
        if (blocks > 65535) blocks = 65355;
    }

    template <typename T> 
    __inline__ __device__ T warp_reduce_sum(T val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        return val;
    }

    template <typename T>
    __inline__ __device__ T warp_reduce_max(T val) {
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            val = max(val, __shfl_down_sync(0xffffffff, val, offset));
        }

        return val;
    }

    template <typename T> 
    __inline__ __device__ T block_reduce_sum(T val) {
        static __shared__ T shared[32];
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;

        val = warp_reduce_sum(val);
    
        if (lane == 0) {
            shared[wid] = val;
        }

        __syncthreads();

        val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

        if (wid == 0) val = warp_reduce_sum(val);

        return val;
    }

    template <typename T>
    __inline__ __device__ T block_reduce_max(T val) {
        static __shared__ T shared[32];
        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;

        val = warp_reduce_max(val);

        if (lane == 0) shared[wid] = val;
        __syncthreads();

        val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;

        if (wid == 0) val = warp_reduce_max(val);

        return val;
    }

    /*******************************************
     *                                         *
     *         GENERAL BINARY KERNEL           *
     *                                         *
     *******************************************/


    #define DEFINE_BINARY_KERNEL(name, op) \
        __global__ void name##_kernel(size_t n, const float* a, const float* b, float* out) { \
            size_t idx = blockIdx.x * blockDim.x + threadIdx.x; \
            for (; idx < n; idx += gridDim.x * blockDim.x) out[idx] = op; \
        } \
        void name##_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept { \
            int b_dim, t_dim; get_launch_config(n, b_dim, t_dim); \
            name##_kernel<<<b_dim, t_dim>>>(n, a, b, out); \
        }

    DEFINE_BINARY_KERNEL(add, a[idx] + b[idx])
    DEFINE_BINARY_KERNEL(sub, a[idx] - b[idx])
    DEFINE_BINARY_KERNEL(mul, a[idx] * b[idx])
    DEFINE_BINARY_KERNEL(div, a[idx] / b[idx])


    /*******************************************
     *                                         *
     *                   FILL                  *
     *                                         *
     *******************************************/


    __global__ void fill_kernel(size_t n, float val, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        for (; idx < n; idx += gridDim.x * blockDim.x) {
            out[idx] = val;
        } 
    }

    void fill_f32(size_t n, float value, float* AXON_RESTRICT out) noexcept {
        int b, t;
        get_launch_config(n, b, t);
        fill_kernel<<<b, t>>> (n, value, out);
    }

    /*******************************************
     *                                         *
     *              SQUARE ROOT                *
     *                                         *
     *******************************************/


    __global__ void sqrt_kernel(size_t n, const float* input, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            out[idx] = sqrtf(input[idx]);
        }
    }

    void sqrt_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        int b, t;
        get_launch_config(n, b, t);
        sqrt_kernel<<<b, t>>> (n, input, output);
    }


    /*******************************************
     *                                         *
     *              EXPONENTIAL                *
     *                                         *
     *******************************************/


    __global__ void exp_kernel(size_t n, const float* input, float* output) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = expf(input[idx]);
        }
    }

    void exp_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        int b, t;
        get_launch_config(n, b, t);
        exp_kernel<<<b, t>>> (n, input, output);
    }

    /*******************************************
     *                                         *
     *                NEGATIVE                 *
     *                                         *
     *******************************************/


    __global__ void neg_kernel(size_t n, const float* input, float* output) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = -input[idx];
        }
    }

    void neg_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        int b, t;
        get_launch_config(n, b, t);
        neg_kernel<<<b, t>>>(n, input, output);
    }

    /*******************************************
     *                                         *
     *                  ReLU                   *
     *                                         *
     *******************************************/
    
    
    __global__ void relu_kernel(size_t n, const float* input, float* output) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            output[idx] = fmaxf(input[idx], 0.0f);
        }
    }
    
    void relu_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) {
        int b, t;
        get_launch_config(n, b, t);
        relu_kernel<<<b, t>>> (n, input, output);
    }
    
    __global__ void relu_backward_kernel(size_t n, const float* input, const float* grad_output, float* grad_input) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
        }
    }
    
    void relu_backward_f32(
        size_t n, const float* AXON_RESTRICT input, 
        const float* AXON_RESTRICT grad_output, 
        float* AXON_RESTRICT grad_input) noexcept {
            
        int b, t;
        get_launch_config(n, b, t);
        relu_backward_kernel<<<b, t>>> (n, input, grad_output, grad_input);
    }

    /*******************************************
     *                                         *
     *                  GELU                   *
     *                                         *
     *******************************************/
  
    __global__ void gelu_kernel(size_t n, const float* input, float* out) {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x = input[idx];
            float x3 = x * x * x;
            float inner = 0.79788456f * (x + 0.044715f * x3);
            out[idx] = 0.5f * x * (1.0f + tanh(inner));
        }
    }

    void gelu_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        int b, t;
        get_launch_config(n, b, t);
        gelu_kernel<<<b, t>>>(n, input, output);
    }

    

} // axon::kernels::gpu
