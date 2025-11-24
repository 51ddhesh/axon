// src/TensorMath.cpp
// github.com/51ddhesh/axon
// MIT License

#include "axon/Tensor.hpp"
#include "axon/ShapeUtils.hpp"
#include <cmath>

namespace axon {

// NDim looper: Iterates over output tensor, 
// mapping coordinates to A and B and handling broadcasting
static void apply_binary_op(Tensor& out, const Tensor& a, const Tensor& b, std::function<double(double, double)> op) {
    
    const auto& out_shape = out.shape();
    size_t n_dim = out_shape.size();

    size_t a_offset = n_dim - a.shape().size();
    size_t b_offset = n_dim - b.shape().size();

    size_t total_elements = out.size();
    std::vector<size_t> coord(n_dim, 0);

    for (size_t i = 0; i < total_elements; i++) {
        std::vector<size_t> a_coord(a.shape().size());
        std::vector<size_t> b_coord(b.shape().size());
    
        for (size_t d = 0; d < a.shape().size(); d++) {
            if (a.shape()[d] == 1) a_coord[d] = 0;
            else a_coord[d] = coord[d + a_offset];
        }

        for (size_t d = 0; d < b.shape().size(); d++) {
            if (b.shape()[d] == 1) b_coord[d] = 0;
            else b_coord[d] = coord[d + b_offset];
        }
    
        double val_a = a(a_coord);
        double val_b = b(b_coord);
        out(coord) = op(val_a, val_b);

        for (int d = n_dim - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < out_shape[d]) break;
            coord[d] = 0;
        }
    }
}

// * OPERATOR IMPLEMENTATIONS
Tensor Tensor::operator+ (const Tensor& other) const {
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(shape_, other.shape_);
    Tensor out(out_shape);
    apply_binary_op(out, *this, other, [](double x, double y) { return x + y; });
    out.prev_.push_back(*this);
    out.prev_.push_back(other);
    return out;
}

Tensor Tensor::operator- (const Tensor& other) const {
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(shape_, other.shape_);
    Tensor out(out_shape);
    apply_binary_op(out, *this, other, [](double x, double y) { return x - y; });
    out.prev_.push_back(*this);
    out.prev_.push_back(other);
    return out;
}

Tensor Tensor::operator* (const Tensor& other) const {
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(shape_, other.shape_);
    Tensor out(out_shape);
    apply_binary_op(out, *this, other, [](double x, double y) { return x * y; });
    out.prev_.push_back(*this);
    out.prev_.push_back(other);
    return out;
}

// Unary Negation: 0 - x = -x
Tensor Tensor::operator- () const {
    Tensor zero = Tensor::zeros(shape_);
    return zero - *this;
}

// * REDUCTIONS
Tensor Tensor::sum() const {
    double total = 0.0;

    // if contiguous, just sum up the raw data 
    if (is_contiguous()) {
        const double* p = data_ptr();
        size_t len = size();
        for (size_t i = 0; i < len; i++) total += p[i];
    } else {
        // Make the Tensor contiguous then sum
        // This is safer than a strided iterator sum
        Tensor c = this -> contiguous();
        return c.sum();
    }

    // Return a zero-dim tensor
    return Tensor({1}, total);
}

} // namespace axon

