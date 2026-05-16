#pragma once

#include "tensor.hpp"

namespace axon {
    Tensor add(Tensor& a, Tensor& b);
    Tensor add(const Tensor& a, const Tensor& b);

    Tensor sub(Tensor& a, Tensor& b);
    Tensor sub(const Tensor& a, const Tensor& b);

    Tensor mul(Tensor& a, Tensor& b);
    Tensor mul(const Tensor& a, const Tensor& b);

    Tensor div(Tensor& a, Tensor& b);
    Tensor div(const Tensor& a, const Tensor& b);

    Tensor neg(Tensor& t);
    Tensor neg(const Tensor& t);

    Tensor sqrt(Tensor& t);
    Tensor sqrt(const Tensor& t);

    Tensor exp(Tensor& t);
    Tensor exp(const Tensor& t);

    Tensor transpose(Tensor& t, int dim0, int dim1);
    Tensor transpose(const Tensor& t, int dim0, int dim1);

    Tensor view(Tensor& t, const std::vector<int>& new_shape);
    Tensor view(const Tensor& t, const std::vector<int>& new_shape);

    Tensor permute(Tensor& t, const std::vector<int>& dims);
    Tensor permute(const Tensor& t, const std::vector<int>& dims);

    Tensor matmul(Tensor& a, Tensor& b);
    Tensor matmul(const Tensor& a, const Tensor& b);

    Tensor sum(Tensor& a);
    Tensor sum(const Tensor& a);

    Tensor sum(Tensor& t, int dim, bool keepdims = false);
    Tensor sum(const Tensor& t, int dim, bool keepdims = false);

    Tensor relu(Tensor& t);
    Tensor relu(const Tensor& t);

    Tensor log_softmax(Tensor& t);
    Tensor log_softmax(const Tensor& t);

    Tensor nll_loss(Tensor& input, Tensor& target);
    Tensor nll_loss(const Tensor& input, const Tensor& target);

    Tensor gelu(Tensor& t);
    Tensor gelu(const Tensor& t);

    Tensor softmax(Tensor& t);
    Tensor softmax(const Tensor& t);

    Tensor embedding(Tensor& input, Tensor& weight);
    Tensor embedding(const Tensor& input, const Tensor& weight);

    Tensor layer_norm(Tensor& input, Tensor& gamma, Tensor& beta, float eps = 1e-5);
    Tensor layer_norm(const Tensor& input, const Tensor& gamma, const Tensor& beta, float eps = 1e-5);

    inline Tensor operator+ (const Tensor& a, const Tensor& b) {
        Tensor a_copy = const_cast<Tensor&>(a);
        Tensor b_copy = const_cast<Tensor&>(b);
        return add(a_copy, b_copy);
    }

    inline Tensor operator- (const Tensor& a, const Tensor& b) {
        Tensor a_copy = const_cast<Tensor&>(a);
        Tensor b_copy = const_cast<Tensor&>(b);
        return sub(a_copy, b_copy);
    }

    inline Tensor operator* (const Tensor& a, const Tensor& b) {
        Tensor a_copy = const_cast<Tensor&>(a);
        Tensor b_copy = const_cast<Tensor&>(b);
        return mul(a_copy, b_copy);
    }

    inline Tensor operator/ (const Tensor& a, const Tensor& b) {
        Tensor a_copy = const_cast<Tensor&>(a);
        Tensor b_copy = const_cast<Tensor&>(b);
        return div(a_copy, b_copy);
    }

    inline Tensor operator- (const Tensor& a) {
        Tensor a_copy = const_cast<Tensor&>(a);
        return neg(a_copy);
    }


} // namespace axon