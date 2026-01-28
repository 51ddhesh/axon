#include "axon/kernels.hpp"    
#include <numeric>
#include <cmath>
#include <limits>
#include <cstring>
#include <algorithm>
#include <immintrin.h> // AVX2 / FMA

namespace axon::kernels::cpu {

    void add_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        size_t i = 0;
        // process 8 floats at a time (8 * 32 = 256 bits)
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
        }

        // residual
        for (; i < n; i++) {
            out[i] = a[i] + b[i];
        }
    }     
    
    void sub_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        size_t i = 0;

        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_sub_ps(va, vb));            
        }

        for (; i < n; i++) {
            out[i] = a[i] - b[i];
        }
    }
    
    void mul_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        size_t i = 0;
        
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
        }
        
        for (; i < n; i++) {
            out[i] = a[i] * b[i];
        }
    }
    
    void div_f32(size_t n, const float* AXON_RESTRICT a, const float* AXON_RESTRICT b, float* AXON_RESTRICT out) noexcept {
        size_t i = 0;
    
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            _mm256_storeu_ps(out + i, _mm256_div_ps(va, vb));
        }

        for (; i < n; i++) {
            out[i] = a[i] / b[i];
        }
    }

    void fill_f32(size_t n, float value, float* AXON_RESTRICT out) noexcept {
        size_t i = 0;
        __m256 v = _mm256_set1_ps(value);

        for (; i + 8 <= n; i += 8) {
            _mm256_storeu_ps(out + i, v);
        }

        for (; i < n; i++) {
            out[i] = value;
        }
    }
    
    void matmul_f32(
        size_t M, size_t N, size_t K,
        const float* AXON_RESTRICT a, 
        const float* AXON_RESTRICT b,
        float* AXON_RESTRICT out) noexcept {

        std::memset(out, 0, M * N * sizeof(float));

        for (size_t i = 0; i < M; i++) {
            for (size_t k = 0; k < K; k++) {
                float val_a = a[i * K + k];
                __m256 va = _mm256_set1_ps(val_a);

                size_t j = 0;

                for (; j + 8 <= N; j += 8) {
                    __m256 vc = _mm256_loadu_ps(&out[i * N + j]);
                    __m256 vb = _mm256_loadu_ps(&b[k * N + j]);
                    // c = c + a * b
                    vc = _mm256_fmadd_ps(va, vb, vc);
                    _mm256_storeu_ps(&out[i * N + j], vc);
                }
                
                for (; j < N; j++) {
                    out[i * N + j] += val_a * b[k * N + j];
                }
            }
        }
    }
        
    
    void sum_f32(size_t n, const float* AXON_RESTRICT inp, float* AXON_RESTRICT out) noexcept {
        float acc = 0.0f;
        for (size_t i = 0; i < n; i++) {
            acc += inp[i];
        }
        *out = acc;
    }

    void relu_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT out) noexcept {
        size_t i = 0;
        __m256 zero = _mm256_setzero_ps();
        for (i; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            _mm256_storeu_ps(out + i, _mm256_max_ps(v, zero));
        }

        for (; i < n; i++) {
            out[i] = input[i] > 0.0f ? input[i] : 0.0f;
        }
    }

    void relu_backward_f32(size_t n, const float* AXON_RESTRICT input, const float* AXON_RESTRICT grad_out, float* AXON_RESTRICT grad_inp) noexcept {
        for (size_t i = 0; i < n; ++i) {
            grad_inp[i] = (input[i] > 0.0f) ? grad_out[i] : 0.0f;
        }
    }

    void log_softmax_f32(size_t rows, size_t cols, const float* __restrict__ input, float* __restrict__ out) noexcept {
        for (size_t r = 0; r < rows; r++) {
            const float* row_input = input + r * cols;
            float* row_out = out + r * cols;

            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < cols; c++) {
                if (row_input[c] > max_val) {
                    max_val = row_input[c];
                }
            }

            float sum_exp = 0.0f;

            for (size_t c = 0; c < cols; c++) {
                float val = std::exp(row_input[c] - max_val);
                row_out[c] = val;
                sum_exp += val;
            }

            float log_sum = std::log(sum_exp);
            
            for (size_t c = 0; c < cols; c++) {
                row_out[c] = (row_input[c] - max_val) - log_sum;
            }
        }
    }

    void log_softmax_backward_f32(size_t rows, size_t cols, 
        const float* __restrict__ grad_output, 
        const float* __restrict__ output, 
        float* __restrict__ grad_input) noexcept {
        // dL/dx_i = dL/dy_i - exp(y_i) * sum(dL/dy_j)
        for (size_t r = 0; r < rows; ++r) {
            const float* grad_row = grad_output + r * cols;
            const float* out_row = output + r * cols;
            float* inp_grad_row = grad_input + r * cols;

            float sum_grad = 0.0f;
            for(size_t c = 0; c < cols; c++) sum_grad += grad_row[c];

            for(size_t c = 0; c < cols; c++) {
                inp_grad_row[c] = grad_row[c] - std::exp(out_row[c]) * sum_grad;
            }
        }
    }

    void sum_dim_f32(size_t outer, size_t dim, size_t inner, const float* __restrict__ input, float* __restrict__ output) noexcept {
        for (size_t i = 0; i < outer * inner; i++) {
            output[i] = 0.0f;
        }

        size_t input_idx, output_idx;

        for (size_t o = 0; o < outer; o++) {
            for (size_t d = 0; d < dim; d++) {
                for (size_t i = 0; i < inner; i++) {
                    input_idx = (o * dim * inner) + (d * inner) + i;
                    output_idx = (o * inner) + i;
                    output[output_idx] += input[input_idx];
                }
            }
        }
    }


    void sqrt_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            _mm256_storeu_ps(output + i, _mm256_sqrt_ps(v));
        }

        for (; i < n; i++) output[i] = std::sqrt(input[i]);
    }
    
    void exp_f32(size_t n, const float* AXON_RESTRICT input, float* AXON_RESTRICT output) noexcept {
        size_t i = 0;
        __m256 zero = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(input + i);
            _mm256_storeu_ps(output + i, _mm256_sub_ps(zero, v));
        }
        for (; i < n; i++) output[i] = -input[i];
    }
    
    void neg_f32(size_t n, const float* input, float* output) noexcept {
        for (size_t i = 0; i < n; i++) {
            output[i] = -(input[i]);
        }
    }

    // GPT-2 uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    void gelu_f32(size_t n, const float* __restrict__ input, float* __restrict__ output) noexcept {
        const float SQRT_2_OVER_PI = 0.79788456080286535587989f;
        const float COEF = 0.044715f;
        
        for (size_t i = 0; i < n; i++) {
            float x = input[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEF * x3);
            float arctan = std::tanh(inner);
            output[i] = 0.5f * x * (1.0f + arctan);
        }
    }
    
    void gelu_backward_f32(size_t n, const float* __restrict__ input, const float* __restrict__ grad_out, float* __restrict__ grad_input) noexcept {
        const float SQRT_2_OVER_PI = 0.79788456080286535587989f;
        const float COEF = 0.044715f;
        
        for (size_t i = 0; i < n; i++) {
            float x = input[i];
            float x3 = x * x * x;
            float inner = SQRT_2_OVER_PI * (x + COEF * x3);
            float tanh_inner = std::tanh(inner);
            
            float secl = 1.0f / (std::cosh(inner));
            float sech2 = secl * secl;

            float dx = 0.5f * (1.0f + tanh_inner) + 
                       0.5f * x * sech2 * SQRT_2_OVER_PI * (1.0f + 3.0f * COEF * x * x);
            
            grad_input[i] = grad_out[i] * dx;
        }
    }

    void embedding_forward_f32(
        size_t vocab_size, size_t dim, size_t num_indices,
        const float* __restrict__ weight, 
        const float* __restrict__ indices, 
        float* __restrict__ out
    ) noexcept {
        for (size_t i = 0; i < num_indices; i++) {
            int idx = static_cast<int>(indices[i]);
            if (idx < 0 || idx >= vocab_size) idx = 0;
            const float* src = weight + idx * dim;
            float* dst = out + i * dim;
            // copy row
            for (size_t r = 0; r < dim; r++) {
                dst[r] = src[r];
            }
        }
    }

    void embedding_backward_f32(
        size_t vocab_size, size_t dim, size_t num_indices,
        const float* __restrict__ grad_output, 
        const float* __restrict__ indices, 
        float* __restrict__ grad_weight
    ) noexcept {
        for (size_t i = 0; i < num_indices; i++) {
            int idx = static_cast<int>(indices[i]);
            if (idx < 0 || idx >= vocab_size) continue;

            const float* g_out_row = grad_output + i * dim;
            float* g_weight_row = grad_weight + idx * dim;

            // scatter add - multiple indices might point to the same row 
            for (size_t r = 0; r < dim; r++) {
                g_weight_row[r] += g_out_row[r];
            }
        }
    }

    void layernorm_forward_f32(
        size_t rows, size_t cols, 
        const float* __restrict__ input,
        const float* __restrict__ gamma, 
        const float* __restrict__ beta,
        float* __restrict__ out, float eps
    ) noexcept {
        for (size_t r = 0; r < rows; r++) {
            const float* in_row = input + r * cols;
            float* out_row = out + r * cols;

            // mean
            float sum = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                sum += in_row[c];
            }

            float mean = sum / cols;

            // variance
            float sum_sq_diff = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                float diff = in_row[c] - mean;
                sum_sq_diff += diff * diff;
            }

            float var = sum_sq_diff / cols;
            float inv_std = 1.0f / std::sqrt(var + eps);

            // normalize 
            for (size_t c = 0; c < cols; c++) {
                float norm = (in_row[c] - mean) * inv_std;
                out_row[c] = norm * gamma[c] + beta[c];
            }
        }
    }

    void layernorm_backward_f32(
        size_t rows, size_t cols,
        const float* __restrict__ grad_out, 
        const float* __restrict__ input,
        const float* __restrict__ gamma, 
        float eps,
        float* __restrict__ grad_input, 
        float* __restrict__ grad_gamma, 
        float* __restrict__ grad_beta
    ) noexcept {
        for (size_t c = 0; c < cols; c++) {
            grad_gamma[c] = 0;
            grad_beta[c] = 0;
        }

        for (size_t r = 0; r < rows; r++) {
            const float* in_row = input + r * cols;
            const float* gout_row = grad_out + r * cols;
            float* gin_row = grad_input + r * cols;

            // recompute the stats
            float sum = 0.0f;
            for (size_t c = 0; c < cols; c++) {
                sum += in_row[c];
            }

            float mean = sum / cols;

            float sum_sq_diff = 0.0f;

            for (size_t c = 0; c < cols; c++) {
                float diff = in_row[c] - mean;
                sum_sq_diff += diff * diff;
            }

            float var = sum_sq_diff / cols;
            float inv_std = 1.0f / std::sqrt(var + eps);

            float sum_dout = 0.0f;
            float sum_dout_xcap = 0.0f;

            for (size_t c = 0; c < cols; c++) {
                float x_cap = (in_row[c] - mean) * inv_std;
                float dy = gout_row[c];

                grad_gamma[c] += dy * x_cap;
                grad_beta[c] += dy;
            
                float dx_cap = dy * gamma[c];
                sum_dout += dx_cap;
                sum_dout_xcap += dx_cap * x_cap;
            }

            float inv_N = 1.0f / cols;
            for (size_t c = 0; c < cols; c++) {
                float x_cap = (in_row[c] - mean) * inv_std;
                float dx_cap = gout_row[c] * gamma[c];
                
                gin_row[c] = inv_N * inv_std * ((cols * dx_cap) - sum_dout - (x_cap * sum_dout_xcap));
            }
        }   
    }

    void softmax_f32(
        size_t rows, size_t cols, 
        const float* __restrict__ input,
        float* __restrict__ out
    ) noexcept {
        for (size_t r = 0; r < rows; r++) {
            const float* in_ptr = input + r * cols;
            float* out_ptr = out + r * cols;

            float max_val = -std::numeric_limits<float>::infinity();
            for (size_t c = 0; c < cols; c++) {
                max_val = std::max(in_ptr[c], max_val);
            }

            float sum = 0.0f;

            for (size_t c = 0; c < cols; c++) {
                float val = std::exp(in_ptr[c] - max_val);
                out_ptr[c] = val;
                sum += val;
            }

            float inv_sum = 1.0f / sum;
            for (size_t c = 0; c < cols; c++) {
                out_ptr[c] *= inv_sum;
            }
        }
    }

    void softmax_backward_f32(
        size_t rows, size_t cols, 
        const float* __restrict__ grad_output, 
        const float* __restrict__ output, 
        float* __restrict__ grad_input
    ) noexcept {
        for (size_t r = 0; r < rows; r++) {
            const float* gout_ptr = grad_output + r * cols;
            const float* out_ptr = output + r * cols;
            float* gin_ptr = grad_input + r * cols;

            float dot = 0.0f;

            for (size_t c = 0; c < cols; c++) {
                dot += gout_ptr[c] * out_ptr[c];
            }

            for (size_t c = 0; c < cols; c++) {
                gin_ptr[c] = out_ptr[c] * (gout_ptr[c] - dot);
            }
        }
    }
}
