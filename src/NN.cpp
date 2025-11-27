#include "axon/NN.hpp"
#include <random>
#include <cmath>
#include <iostream>

namespace axon::nn {

    Linear::Linear(size_t in_features, size_t out_features, bool bias) 
        : use_bias(bias) {
        
        // 1. Allocate shapes
        // We use (In, Out) for weights so we can do Input(Batch, In) @ W(In, Out)
        W = Tensor::zeros({in_features, out_features});
        if (use_bias) {
            b = Tensor::zeros({1, out_features});
        }

        // 2. Initialize Weights (Kaiming / He Uniform)
        reset_parameters(in_features, out_features);
    }

    void Linear::reset_parameters(size_t in, size_t) { 
        std::random_device rd;
        std::mt19937 gen(rd());

        double bound = 1.0 / std::sqrt((double)in); 
        std::uniform_real_distribution<> dis(-bound, bound);

        double* w_ptr = W.data_ptr();
        for (size_t i = 0; i < W.size(); ++i) {
            w_ptr[i] = dis(gen);
        }

        // Initialize Bias to 0.0
        if (use_bias) {
            double* b_ptr = b.data_ptr();
            for (size_t i = 0; i < b.size(); ++i) b_ptr[i] = 0.0;
        }
    }

    Tensor Linear::operator()(const Tensor& input) {
        // Linear: X @ W + b
        Tensor out = input.matmul(W);
        if (use_bias) {
            out = out + b;
        }
        return out;
    }

    std::vector<Tensor> Linear::parameters() {
        if (use_bias) return {W, b};
        return {W};
    }

} // namespace axon::nn