#pragma once
#include "tensor.hpp"
#include <vector>
#include <cmath>

namespace axon {
    class SGD {
        std::vector<Tensor> parameters;
        float lr;

    public:
        SGD(std::vector<Tensor> params, float learning_rate) 
            : parameters(params), lr(learning_rate) {}

        void zero_grad() {
            for (auto& p : parameters) {
                p.zero_grad();
            }
        }

        void step() {
            for (auto& p : parameters) {
                if (p.get_grad()) {
                    // p.data -= lr * p.grad.data
                    // We access raw pointers for speed and to avoid graph tracking
                    Tensor g_c = p.get_grad() -> is_contiguous() ? *p.get_grad() : p.get_grad() -> contiguous();
                    size_t n = p.numel();
                    float* p_ptr = p.data_ptr();
                    const float* g_ptr = g_c.data_ptr();

                    // Simple update loop (AVX candidate later)
                    for (size_t i = 0; i < n; ++i) {
                        p_ptr[i] -= lr * g_ptr[i];
                    }
                }
            }
        }
    };

    class AdamW {
        struct ParamState {
            std::vector<float> m;
            std::vector<float> v;
        };

        std::vector<Tensor> parameters;
        std::vector<ParamState> states;
        float lr, beta_1, beta_2, eps, weight_decay;
        int t;

    public:
        AdamW(
            std::vector<Tensor> params, float learning_rate = 1e-3,
            float beta_1 = 0.9, float beta_2 = 0.999, float eps = 1e-8, float weight_decay = 0.01
        ) : parameters(params), lr(learning_rate), 
            beta_1(beta_1), beta_2(beta_2), eps(eps), 
            weight_decay(weight_decay), t(0) {
            
            for (const auto& p : parameters) {
                size_t n = p.numel();
                states.push_back({std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f)});
            }
        }

        void zero_grad() {
            for (auto& p : parameters) {
                p.zero_grad();
            }
        }

        void step() {
            t++;
            float bias_correction1 = 1.0f - std::pow(beta_1, t);
            float bias_correction2 = 1.0f - std::pow(beta_2, t);

            size_t param_size = parameters.size();

            for (size_t i = 0; i < param_size; i++) {
                Tensor& p = parameters[i];
                if (!p.get_grad()) {
                    continue;
                }

                Tensor g_tensor = p.get_grad() -> is_contiguous() ? *p.get_grad() : p.get_grad() -> contiguous();

                float* p_ptr = p.data_ptr();
                const float* g_ptr = g_tensor.data_ptr();

                std::vector<float>& m = states[i].m;
                std::vector<float>& v = states[i].v;
                size_t n = p.numel();

                // CPU Loop (Compiler will auto-vectorize this if flags are set)
                for (size_t j = 0; j < n; ++j) {
                    float grad = g_ptr[j];

                    // Weight Decay (decoupled)
                    p_ptr[j] -= lr * weight_decay * p_ptr[j];

                    // Update moments
                    m[j] = beta_1 * m[j] + (1.0f - beta_1) * grad;
                    v[j] = beta_2 * v[j] + (1.0f - beta_2) * grad * grad;

                    // Bias correction
                    float m_cap = m[j] / bias_correction1;
                    float v_cap = v[j] / bias_correction2;

                    // Update parameter
                    p_ptr[j] -= lr * m_cap / (std::sqrt(v_cap) + eps);
                }
            }
        }
    };
}