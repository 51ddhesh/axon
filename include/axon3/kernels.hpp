#pragma once 

#include <cstddef>

namespace axon::kernels {
    void add_f32(size_t n, const float* a, const float* b, float* out) noexcept;    
    void sub_f32(size_t n, const float* a, const float* b, float* out) noexcept;
    void mul_f32(size_t n, const float* a, const float* b, float* out) noexcept;
    void div_f32(size_t n, const float* a, const float* b, float* out) noexcept;
    
    void matmul_f32(size_t M, size_t N, size_t K, const float* a, const float* b, float* out) noexcept;
    void relu_f32(size_t n, const float* input, float* out) noexcept;
    void relu_backward_f32(size_t n, const float* input, const float* grad_out, float* grad_input) noexcept;
    void log_softmax_f32(size_t rows, size_t cols, const float* input, float* out) noexcept;
    void log_softmax_backward_f32(size_t rows, size_t cols, const float* grad_output, const float* output, float* grad_input) noexcept;
    void gelu_f32(size_t n, const float* input, float* output) noexcept;
    void gelu_backward_f32(size_t n, const float* input, const float* grad_out, float* grad_input) noexcept;
    
    void softmax_f32(size_t rows, size_t cols, const float* input, float* out) noexcept;
    void softmax_backward_f32(size_t rows, size_t cols, const float* grad_output, const float* output, float* grad_input) noexcept;

    void sum_f32(size_t n, const float* inp, float* out) noexcept;
    void sum_dim_f32(size_t outer, size_t dim, size_t inner, const float* input, float* output) noexcept;
    
    void sqrt_f32(size_t n, const float* input, float* output) noexcept;
    void exp_f32(size_t n, const float* input, float* output) noexcept;
    void neg_f32(size_t n, const float* input, float* output) noexcept;
    
    void embedding_forward_f32(
        size_t vocab_size, size_t dim, size_t num_indices,
        const float* weight, const float* indices, float* out
    ) noexcept;
    void embedding_backward_f32(
        size_t vocab_size, size_t dim, size_t num_indices,
        const float* grad_output, const float* indices, float* grad_weight
    ) noexcept;

    void layernorm_forward_f32(
        size_t rows, size_t cols, const float* input,
        const float* gamma, const float* beta,
        float* out, float eps
    ) noexcept;
    void layernorm_backward_f32(
        size_t rows, size_t cols,
        const float* grad_out, const float* input,
        const float* gamma, float eps,
        float* grad_input, float* grad_gamma, float* grad_beta
    ) noexcept;
} // namespace axon::kernels
