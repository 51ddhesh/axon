// src/TensorMath.cpp
// github.com/51ddhesh/axon
// MIT License

#include "axon/Tensor.hpp"
#include "axon/ShapeUtils.hpp"
#include <cmath>
#include <functional>
#include <iostream>

namespace axon {

// NDim looper: Iterates over output tensor, 
// mapping coordinates to A and B and handling broadcasting
static void apply_binary_op(Tensor& out, const Tensor& a, const Tensor& b, 
                            std::function<double(double, double)> op) {
    
    const auto& out_shape = out.shape();
    size_t ndim = out_shape.size();
    
    std::vector<size_t> coord(ndim, 0);
    size_t total_elements = out.size();
    
    size_t a_offset = ndim - a.shape().size();
    size_t b_offset = ndim - b.shape().size();
    
    for (size_t i = 0; i < total_elements; ++i) {
        std::vector<size_t> a_coord(a.shape().size());
        std::vector<size_t> b_coord(b.shape().size());
        
        for (size_t d = 0; d < a.shape().size(); ++d) {
            a_coord[d] = (a.shape()[d] == 1) ? 0 : coord[d + a_offset];
        }
        for (size_t d = 0; d < b.shape().size(); ++d) {
            b_coord[d] = (b.shape()[d] == 1) ? 0 : coord[d + b_offset];
        }
        
        out(coord) = op(a(a_coord), b(b_coord));
        
        for (int d = ndim - 1; d >= 0; --d) {
            coord[d]++;
            if (coord[d] < out_shape[d]) break;
            coord[d] = 0;
        }
    }
}


// * AUTOGRAD HELPERS (UNBROADCASTING DURING BACKWARD())

static Tensor unbroadcast(const Tensor& grad, const std::vector<size_t>& target_shape) {
    Tensor out = grad;

    // Handle the rank difference
    while (out.shape().size() > target_shape.size()) {
        out = out.sum(0);
    }

    // Handle dim mismatch
    /*
        If grad = (32, 10), target = (1, 10)
        (32, 10) => sum(0) => (10, ) => reshape(1, 10)
    */
    // sum over dims where target == 1 and grad > 1
    for (size_t i = 0; i < target_shape.size(); i++) {
        if (target_shape[i] == 1 && out.shape()[i] > 1) {
            std::vector<size_t> kept_shape = out.shape();
            kept_shape[i] = 1;
            out = out.sum(i).reshape(kept_shape);
        }
    }
    return out;
}

Tensor Tensor::sum(size_t dim) const {
    if (dim >= shape_.size()) {
        throw std::runtime_error("Dimension out of range");
    }

    std::vector<size_t> new_shape = shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor out = (new_shape.empty() ? Tensor({1}, 0.0) : Tensor::zeros(new_shape));

    size_t total = size();
    std::vector<size_t> in_coord(shape_.size(), 0);

    for (size_t i = 0; i < total; i++) {
        std::vector<size_t> out_coord;
        out_coord.reserve(shape_.size() - 1);
        for (size_t d = 0; d < shape_.size(); d++) {
            if (d != dim) out_coord.push_back(in_coord[d]);
        }
        double val = (*this)(in_coord);

        if (out_coord.empty()) {
            out_coord = {0};
        }

        out(out_coord) += val;

        for (int d = shape_.size() - 1; d >= 0; d--) {
            in_coord[d]++;
            if (in_coord[d] < shape_[d]) break;
            in_coord[d] = 0;
        }
    }
    return out;
}


// * OPERATIONS

// ! ONLY SUPPORTS 2 DIMENSIONAL TENSORS
// Matrix Multiplication
Tensor Tensor::matmul(const Tensor& other) const {
    // TODO: Add N-Dim support for matmul
    if (shape_.size() != 2 || other.shape_.size() != 2) {
        throw std::runtime_error("Only 2D support available for now");
    }

    // A: I x J, B: J x K, C: I x K (C = A x B)
    if (shape_[1] != other.shape_[0]) {
        throw std::runtime_error("Matrix shapes mismatch");
    }

    size_t M = shape_[0];
    size_t K = shape_[1];
    size_t N = other.shape_[1];

    Tensor out({M, N}, 0.0);

    Tensor A = is_contiguous() ? *this : this -> contiguous();
    Tensor B = other.is_contiguous() ? other : other.contiguous();

    const double* a_ptr = A.data_ptr();
    const double* b_ptr = B.data_ptr();
    double* out_ptr = out.data_ptr();

    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                sum += a_ptr[m * K + k] * b_ptr[k * N + n];
            }
            out_ptr[m * N + n] = sum;
        }
    }

    out.prev_.push_back(*this);
    out.prev_.push_back(other);

    Tensor self = *this;
    Tensor rhs = other;

    out.set_backward_fn([out, self, rhs]() mutable {
        /*
            Gradient formula:
            C = A @ B
            dl/dA = dl/dC @ B.T
            dl/dB = A.T @ dl/dC
        */

        Tensor grad_out({out.shape()[0], out.shape()[1]}, 0.0);
        std::copy(out.grad_ptr(), out.grad_ptr() + out.size(), grad_out.data_ptr());
        
        // Calculate dL/dA
        Tensor B_T = rhs.transpose(0, 1);
        Tensor d_self = grad_out.matmul(B_T); 
        
        // Accumulate into self.grad
        if (!d_self.is_contiguous()) {
            d_self = d_self.contiguous();
        } 
        
        double* dst = self.grad_ptr();
        double* src = d_self.data_ptr();

        size_t len = self.size();
        for(size_t i = 0; i < len; i++) {
            dst[i] += src[i];
        }
        
        // Calculate dL/dB
        Tensor A_T = self.transpose(0, 1);
        Tensor d_rhs = A_T.matmul(grad_out);
        
        // Accumulate into rhs.grad
        if (!d_rhs.is_contiguous()) {
            d_rhs = d_rhs.contiguous();
        } 
        
        dst = rhs.grad_ptr();
        src = d_rhs.data_ptr();

        len = rhs.size();
        for(size_t i = 0; i < len; i++) {
            dst[i] += src[i];
        } 
    });

    return out;
}



// * OPERATOR IMPLEMENTATIONS

Tensor Tensor::operator+ (const Tensor& other) const {
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(shape_, other.shape_);
    Tensor out(out_shape);
    apply_binary_op(out, *this, other, [](double x, double y) { return x + y; });

    out.prev_.push_back(*this);
    out.prev_.push_back(other);

    Tensor self = *this;
    Tensor rhs = other;

    out.set_backward_fn([out, self, rhs]() mutable {
        Tensor grad = Tensor(out.shape());
        Tensor grad_out = Tensor(out.shape());
        std::copy(out.grad_ptr(), out.grad_ptr() + out.size(), grad_out.data_ptr());

        Tensor grad_self = unbroadcast(grad_out, self.shape());
    
        double* self_g = self.grad_ptr();
        if (!grad_self.is_contiguous()) {
            grad_self = grad_self.contiguous();
        }
        double* src_g = grad_self.data_ptr();

        for (size_t i = 0; i < self.size(); i++) {
            self_g[i] += src_g[i];
        }

        Tensor grad_rhs = unbroadcast(grad_out, rhs.shape());
        if (!grad_rhs.is_contiguous()) {
            grad_rhs = grad_rhs.contiguous();
        }
        src_g = grad_rhs.data_ptr();
        double* rhs_g = rhs.grad_ptr();
        for (size_t i = 0; i < rhs.size(); i++) {
            rhs_g[i] += src_g[i];
        }
    }); 

    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(shape_, other.shape_);
    Tensor out(out_shape);
    apply_binary_op(out, *this, other, [](double x, double y) { return x - y; });
    
    out.prev_.push_back(*this);
    out.prev_.push_back(other);

    Tensor self = *this;
    Tensor rhs = other;

    out.set_backward_fn([out, self, rhs]() mutable {
        Tensor grad_out = Tensor(out.shape());
        std::copy(out.grad_ptr(), out.grad_ptr() + out.size(), grad_out.data_ptr());

        // Self: +1 * grad
        Tensor grad_self = unbroadcast(grad_out, self.shape());
    
        if (!grad_self.is_contiguous()) {
            grad_self = grad_self.contiguous();
        } 
    
        double* s_ptr = grad_self.data_ptr();
        double* self_g = self.grad_ptr();
        
        for(size_t i = 0; i < self.size(); i++) {
            self_g[i] += s_ptr[i];
        } 

        // RHS: -1 * grad
        Tensor grad_rhs = unbroadcast(grad_out, rhs.shape());

        if (!grad_rhs.is_contiguous()) {
            grad_rhs = grad_rhs.contiguous();
        } 

        double* r_ptr = grad_rhs.data_ptr();
        double* rhs_g = rhs.grad_ptr();
        for(size_t i = 0; i < rhs.size(); i++) {
            rhs_g[i] -= r_ptr[i]; 
        } 
    });

    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(shape_, other.shape_);
    Tensor out(out_shape);
    apply_binary_op(out, *this, other, [](double x, double y) { return x * y; });
    
    out.prev_.push_back(*this);
    out.prev_.push_back(other);

    Tensor self = *this;
    Tensor rhs = other;

    out.set_backward_fn([out, self, rhs]() mutable {
        Tensor grad_out = Tensor(out.shape());
        std::copy(out.grad_ptr(), out.grad_ptr() + out.size(), grad_out.data_ptr());

        // dL/dSelf = Grad_Out * RHS
        // Construct broadcasted RHS values
        Tensor rhs_broad(out.shape());
        // Use apply_binary_op to fill it? No, just use operator* logic.
        
        // Calculate Term for Self: (Grad * RHS) -> Unbroadcast
        Tensor term1(out.shape());
        apply_binary_op(term1, grad_out, rhs, [](double g, double r){ return g * r; });
        Tensor grad_self = unbroadcast(term1, self.shape());
        
        if (!grad_self.is_contiguous()) {
            grad_self = grad_self.contiguous();
        } 

        double* src = grad_self.data_ptr();
        double* dst = self.grad_ptr();
        
        for(size_t i = 0; i < self.size(); i++) {
            dst[i] += src[i];
        }

        // Calculate Term for RHS: (Grad * Self) -> Unbroadcast
        Tensor term2(out.shape());
        apply_binary_op(term2, grad_out, self, [](double g, double s){ return g * s; });
        Tensor grad_rhs = unbroadcast(term2, rhs.shape());

        if (!grad_rhs.is_contiguous()) {
            grad_rhs = grad_rhs.contiguous();
        }
        
        src = grad_rhs.data_ptr();
        dst = rhs.grad_ptr();
        
        for(size_t i = 0; i < rhs.size(); i++) {
            dst[i] += src[i];
        } 
    });

    return out;
}


Tensor Tensor::operator-() const {
    Tensor zero = Tensor::zeros(shape_);
    return zero - *this;
}

// * REDUCTIONS

Tensor Tensor::sum() const {
    double total = 0.0;
    
    if (is_contiguous()) {
        const double* ptr = data_ptr();
        size_t len = size();
        for(size_t i=0; i<len; ++i) total += ptr[i];
    } else {
        Tensor c = this -> contiguous();
        return c.sum(); 
    }
    
    // Result is a scalar (1-element vector)
    Tensor out({1}, total);
    
    // 1. Connect Graph
    out.prev_.push_back(*this);
    
    // 2. Backward
    Tensor self = *this;
    out.set_backward_fn([out, self]() mutable {
        // Sum gradient: distributes the scalar gradient to ALL elements
        // dL/dx_i = dL/dSum * 1.0
        double grad_val = out.grad_ptr()[0];
        double* self_g = self.grad_ptr();
        size_t n = self.size();
        
        //  assume standard tensor.
        for(size_t i=0; i<n; ++i) {
            self_g[i] += grad_val;
        }
    });
    
    return out;
}

} // namespace axon