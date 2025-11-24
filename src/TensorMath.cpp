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

// * OPERATOR IMPLEMENTATIONS

Tensor Tensor::operator+(const Tensor& other) const {
    std::vector<size_t> out_shape = shape_utils::broadcast_shapes(shape_, other.shape_);
    Tensor out(out_shape);
    apply_binary_op(out, *this, other, [](double x, double y) { return x + y; });
    
    // 1. Graph Connectivity
    out.prev_.push_back(*this);
    out.prev_.push_back(other);

    // 2. Backward Function (Capture by Value)
    Tensor self = *this;
    Tensor rhs = other;
    
    out.set_backward_fn([out, self, rhs]() mutable {
        // Simple Gradient Flow (Naive: Assumes shapes match)
        double* out_g = out.grad_ptr();
        double* self_g = self.grad_ptr();
        double* rhs_g = rhs.grad_ptr();
        size_t n = out.size();

        // Check for Broadcasting 
        if (out.shape() == self.shape() && out.shape() == rhs.shape()) {
            for (size_t i = 0; i < n; ++i) {
                self_g[i] += out_g[i]; // dL/dx += dL/dout * 1
                rhs_g[i] += out_g[i];  // dL/dy += dL/dout * 1
            }
        } else {
            // Fallback for now: just warn
             std::cerr << "Warning: Broadcast backward not implemented in operator+ yet!" << std::endl;
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
        double* out_g = out.grad_ptr();
        double* self_g = self.grad_ptr();
        double* rhs_g = rhs.grad_ptr();
        size_t n = out.size();

        if (out.shape() == self.shape() && out.shape() == rhs.shape()) {
            for (size_t i = 0; i < n; ++i) {
                self_g[i] += out_g[i];       // dL/dx += dL/dout * 1
                rhs_g[i] += -1.0 * out_g[i]; // dL/dy += dL/dout * (-1)
            }
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
        // Product Rule: d(uv) = udv + vdu
        // dL/du = dL/dout * v
        // dL/dv = dL/dout * u
        
        // We need VALUES of self and rhs, not just gradients
        // Since we don't have easy iterators yet, we assume contiguous for this simple test
        // or use operator(). For speed/simplicity in Phase 4 Step 1, assuming contiguous.
        
        if (out.is_contiguous() && self.is_contiguous() && rhs.is_contiguous() &&
            out.shape() == self.shape() && out.shape() == rhs.shape()) {
                
            double* out_g = out.grad_ptr();
            double* self_g = self.grad_ptr();
            double* rhs_g = rhs.grad_ptr();
            const double* self_d = self.data_ptr();
            const double* rhs_d = rhs.data_ptr();
            size_t n = out.size();
            
            for(size_t i=0; i<n; ++i) {
                self_g[i] += out_g[i] * rhs_d[i];
                rhs_g[i] += out_g[i] * self_d[i];
            }
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