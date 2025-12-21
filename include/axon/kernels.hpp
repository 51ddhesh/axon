#pragma once 

#include <cstddef>

#if defined(_MSC_VER)
    #define AXON_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define AXON_RESTRICT __restrict__
#else
    #define AXON_RESTRICT
#endif

namespace axon::kernels {
    // Element-wise ops
    void add_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept;    
    void sub_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept;
    void mul_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept;
    void div_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept;
    
    // Matrix Multiplication
    void matmul_f32(size_t M, size_t N, size_t K, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept;
    
    // Activation & Others
    void relu_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) noexcept;
    void relu_backward_f32(size_t n, const float* AXON_RESTRICT input, const float* AXON_RESTRICT grad_out, float* AXON_RESTRICT grad_input) noexcept;
    
    void log_softmax_f32(size_t rows, size_t cols, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) noexcept;
    void log_softmax_backward_f32(size_t rows, size_t cols, const float* AXON_RESTRICT grad_output, const float* AXON_RESTRICT output, float* AXON_RESTRICT grad_input) noexcept;
    
    void gelu_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept;
    void gelu_backward_f32(size_t n, const float* AXON_RESTRICT input, const float* AXON_RESTRICT grad_out, float* AXON_RESTRICT grad_input) noexcept;
    
    void softmax_f32(size_t rows, size_t cols, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) noexcept;
    void softmax_backward_f32(size_t rows, size_t cols, const float* AXON_RESTRICT grad_output, const float* AXON_RESTRICT output, float* AXON_RESTRICT grad_input) noexcept;

    void sum_f32(size_t n, const float* AXON_RESTRICT inp, float* AXON_RESTRICT out) noexcept;
    void sum_dim_f32(size_t outer, size_t dim, size_t inner, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept;
    
    void sqrt_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept;
    void exp_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept;
    void neg_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept;
    
    // Embeddings & Norms
    void embedding_forward_f32(
        size_t vocab_size, size_t dim, size_t num_indices,
        const float* AXON_RESTRICT weight, const float* AXON_RESTRICT indices, float* AXON_RESTRICT out
    ) noexcept;
    void embedding_backward_f32(
        size_t vocab_size, size_t dim, size_t num_indices,
        const float* AXON_RESTRICT grad_output, const float* AXON_RESTRICT indices, float* AXON_RESTRICT grad_weight
    ) noexcept;

    void layernorm_forward_f32(
        size_t rows, size_t cols, const float* AXON_RESTRICT input,
        const float* AXON_RESTRICT gamma, const float* AXON_RESTRICT beta,
        float* AXON_RESTRICT out, float eps
    ) noexcept;
    void layernorm_backward_f32(
        size_t rows, size_t cols,
        const float* AXON_RESTRICT grad_out, const float* AXON_RESTRICT input,
        const float* AXON_RESTRICT gamma, float eps,
        float* AXON_RESTRICT grad_input, float* AXON_RESTRICT grad_gamma, float* AXON_RESTRICT grad_beta
    ) noexcept;

} // namespace axon::kernels