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
            Tensor m;
            Tensor v;
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
                Device dev = p.device();
                states.push_back({
                    Tensor::zeros(p.get_shape(), dev),
                    Tensor::zeros(p.get_shape(), dev)
                });
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

                Tensor& g_tensor = *p.get_grad();
                Tensor& m = states[i].m;
                Tensor& v = states[i].v;

                size_t n = p.numel();
                float* p_ptr = p.data_ptr();
                const float* g_ptr = g_tensor.data_ptr();
                float* m_ptr = m.data_ptr();
                float* v_ptr = v.data_ptr();

                Device dev = p.device();
                if (dev.type == DeviceType::CPU) {
                    for (size_t j = 0; j < n; ++j) {
                        float grad = g_ptr[j];

                        p_ptr[j] -= lr * weight_decay * p_ptr[j];

                        m_ptr[j] = beta_1 * m_ptr[j] + (1.0f - beta_1) * grad;
                        v_ptr[j] = beta_2 * v_ptr[j] + (1.0f - beta_2) * grad * grad;

                        float m_cap = m_ptr[j] / bias_correction1;
                        float v_cap = v_ptr[j] / bias_correction2;

                        p_ptr[j] -= lr * m_cap / (std::sqrt(v_cap) + eps);
                    }
                } else {
                    Tensor m_cpu = m.to(Device(DeviceType::CPU));
                    Tensor v_cpu = v.to(Device(DeviceType::CPU));
                    Tensor p_cpu = p.to(Device(DeviceType::CPU));
                    Tensor g_cpu = g_tensor.to(Device(DeviceType::CPU));

                    float* m_cpu_ptr = m_cpu.data_ptr();
                    float* v_cpu_ptr = v_cpu.data_ptr();
                    float* p_cpu_ptr = p_cpu.data_ptr();
                    const float* g_cpu_ptr = g_cpu.data_ptr();

                    for (size_t j = 0; j < n; ++j) {
                        float grad = g_cpu_ptr[j];

                        p_cpu_ptr[j] -= lr * weight_decay * p_cpu_ptr[j];

                        m_cpu_ptr[j] = beta_1 * m_cpu_ptr[j] + (1.0f - beta_1) * grad;
                        v_cpu_ptr[j] = beta_2 * v_cpu_ptr[j] + (1.0f - beta_2) * grad * grad;

                        float m_cap = m_cpu_ptr[j] / bias_correction1;
                        float v_cap = v_cpu_ptr[j] / bias_correction2;

                        p_cpu_ptr[j] -= lr * m_cap / (std::sqrt(v_cap) + eps);
                    }

                    cudaMemcpy(p.data_ptr(), p_cpu.data_ptr(), n * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(m.data_ptr(), m_cpu.data_ptr(), n * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(v.data_ptr(), v_cpu.data_ptr(), n * sizeof(float), cudaMemcpyHostToDevice);
                }
            }
        }
    };
}