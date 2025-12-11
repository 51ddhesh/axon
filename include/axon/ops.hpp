#pragma once 

#include "tensor.hpp"

namespace axon {
    Tensor add(Tensor a, Tensor b);
    Tensor sub(Tensor a, Tensor b);
    Tensor mul(Tensor a, Tensor b);
    Tensor div(Tensor a, Tensor b);

    Tensor neg(Tensor t);
    Tensor sqrt(Tensor t);
    Tensor exp(Tensor t);

    Tensor transpose(Tensor t, int dim0, int dim1);
    Tensor view(Tensor t, const std::vector<int>& new_shape);
    Tensor permute(Tensor t, const std::vector<int>& dims);

    Tensor matmul(Tensor a, Tensor b);
    Tensor sum(Tensor a);
    Tensor sum(Tensor t, int dim, bool keepdims = false);

    Tensor relu(Tensor t);

    Tensor log_softmax(Tensor t);
    // negative log likelihood
    Tensor nll_loss(Tensor input, Tensor target); 

    Tensor gelu(Tensor t);

    Tensor softmax(Tensor t);

    // Embedding: Look up indices in weight
    // Input: (B, T) or (N). Weight: (Vocab, Dim). Output: (B, T, Dim)
    Tensor embedding(Tensor input, Tensor weight); 

    Tensor layer_norm(Tensor input, Tensor gamma, Tensor beta, float eps = 1e-5);

    inline Tensor operator+ (const Tensor& a, const Tensor& b) {
        return add(a, b);
    }

    inline Tensor operator- (const Tensor& a, const Tensor& b) {
        return sub(a, b);
    }

    inline Tensor operator* (const Tensor& a, const Tensor& b) {
        return mul(a, b);
    }

    inline Tensor operator/ (const Tensor& a, const Tensor& b) {
        return div(a, b);
    }

    inline Tensor operator- (const Tensor& a) {
        return neg(a);
    }


} // namespace axon
