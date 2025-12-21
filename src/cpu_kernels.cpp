#include "axon/kernels.hpp"    
#include <numeric>
#include <cmath>
#include <limits>

namespace axon::kernels::cpu {
    
    void add_f32(size_t n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out) noexcept {
        for (size_t i = 0; i < n; i++) {
            out[i] = a[i] + b[i];
        }
    }     
    
    void sub_f32(size_t n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out) noexcept {
        for (size_t i = 0; i < n; i++) {
            out[i] = a[i] - b[i];
        }
    }
    
    void mul_f32(size_t n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out) noexcept {
        for (size_t i = 0; i < n; i++) {
            out[i] = a[i] * b[i];
        }
    }

    void matmul_f32(
        size_t M, size_t N, size_t K, 
        const float* __restrict__ a, 
        const float* __restrict__ b, 
        float* __restrict__ out) noexcept {
        
        for (size_t i = 0; i < M * N; ++i) out[i] = 0.0f;

        // Cache-friendly loop 
        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                float val_a = a[i * K + k];
                for (size_t j = 0; j < N; ++j) {
                    out[i * N + j] += val_a * b[k * N + j];
                }
            }
        }
    }

    void sum_f32(size_t n, const float* __restrict__ inp, float* __restrict__ out) noexcept {
        float acc = 0.0f;
        for (size_t i = 0; i < n; i++) {
            acc += inp[i];
        }
        *out = acc;
    }

    void relu_f32(size_t n, const float* __restrict__ input, float* __restrict__ out) noexcept {
        for (size_t i = 0; i < n; ++i) {
            out[i] = input[i] > 0.0f ? input[i] : 0.0f;
        }
    }

    
    void relu_backward_f32(size_t n, const float* __restrict__ input, const float* __restrict__ grad_out, float* __restrict__ grad_inp) noexcept {
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

    void div_f32(size_t n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ out) noexcept {
        for (size_t i = 0; i < n; i++) {
            out[i] = a[i] / b[i];
        }
    }

    void sqrt_f32(size_t n, const float* __restrict__ input, float* __restrict__ output) noexcept {
        for (size_t i = 0; i < n; i++) {
            output[i] = std::sqrt(input[i]);
        }
    }
    
    void exp_f32(size_t n, const float* __restrict__ input, float* __restrict__ output) noexcept {
        for (size_t i = 0; i < n; i++) {
            output[i] = std::exp(input[i]);
        }
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
