// src/tensor_ops.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Tensor.hpp"
#include "../../include/private/OperationHelpers.hpp"
#include "../../include/TensorMath.hpp"

Tensor operator+ (const Tensor& a, const Tensor& b) {
    auto forward_op = std::plus<axon::dtype::f64>();
    Tensor result = axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, forward_op);
    result._prev = { const_cast<Tensor*>(&a), const_cast<Tensor*>(&b) };

    result._backward_fn = [pa = &a, pb = &b](Tensor* self) mutable {
        // c = a + b
        // dc/da = 1
        // dc/db = 1
        // Apply chain rule 
        // For addition, the gradient distributes
        *(pa -> _grad) += axon::private_helpers::other::sum_to_shape(*(self -> _grad), pa -> getShape());
        *(pb -> _grad) += axon::private_helpers::other::sum_to_shape(*(self -> _grad), pb -> getShape());
    };
    return result;
}

Tensor Tensor::operator+= (const Tensor& other_) {
    (*this) = (*this) + other_;
    return (*this);
}

Tensor operator- (const Tensor& a, const Tensor& b) {
    auto forward_op = std::minus<axon::dtype::f64>();
    Tensor result = axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, forward_op);
    result._prev = { const_cast<Tensor*>(&a), const_cast<Tensor*>(&b) };

    result._backward_fn = [pa = &a, pb = &b](Tensor* self) mutable {
        // c = a - b
        // dc / da = 1
        // dc / db = -1

        auto grad_b = *(self -> _grad) * -1;

        *(pa -> _grad) += axon::private_helpers::other::sum_to_shape(*(self -> _grad), pa -> getShape());
        *(pb -> _grad) += axon::private_helpers::other::sum_to_shape(grad_b, pb -> getShape());
    };
    return result;
}

Tensor Tensor::operator-= (const Tensor& other_) {
    (*this) = (*this) - other_;
    return (*this);
}

Tensor operator* (const Tensor& a, const Tensor& b) {
    auto forward_op = std::multiplies<axon::dtype::f64>();
    Tensor result = axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, forward_op);
    result._prev = { const_cast<Tensor*>(&a), const_cast<Tensor*>(&b) };

    result._backward_fn = [pa = &a, pb = &b](Tensor* self) mutable {
        // c = a * b
        // dc / da = b
        // dc / db = a
        // Applu chain rule
        auto grad_a = *(self -> _grad) * (*pb);
        auto grad_b = *(self -> _grad) * (*pa);

        *(pa -> _grad) += axon::private_helpers::other::sum_to_shape(grad_a, pa -> getShape());
        *(pb -> _grad) += axon::private_helpers::other::sum_to_shape(grad_b, pb -> getShape());
    };

    return result;
}

Tensor Tensor::operator*= (const Tensor& other_) {
    (*this) = (*this) * other_;
    return (*this);
}

// ! NOTE: `std::divides` does not have an in-built assert to check for the divisor being zero
// * Dividing by zero will result in `inf`
Tensor operator/ (const Tensor& a, const Tensor& b) {
    auto forward_op = std::divides<axon::dtype::f64>();
    Tensor result = axon::private_helpers::binary_op_helper::_apply_binary_op(a, b, forward_op);
    result._prev = { const_cast<Tensor*>(&a), const_cast<Tensor*>(&b) };

    result._backward_fn = [pa = &a, pb = &b](Tensor* self) mutable {
        // c = a / b = a * (b ^ -1)
        // dc / da = b ^ -1 = 1 / b
        // dc / db = -a / b ^ 2;
    
        auto grad_a = *(self -> _grad) / (*pb);
        auto grad_b = *(self -> _grad) * (-1 * (*pa) / axon::math::pow((*pb), 2.0));

        *(pa -> _grad) += axon::private_helpers::other::sum_to_shape(grad_a, pa -> getShape());
        *(pb -> _grad) += axon::private_helpers::other::sum_to_shape(grad_b, pb -> getShape());
    };

    return result;
}

Tensor Tensor::operator/= (const Tensor& other_) {
    (*this) = (*this) / other_;
    return (*this);
}

// Tensor Ops with Scalars

// Scalar Addition
Tensor Tensor::operator+ (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] + val_;
    }
    return result;
}

// Scalar Subtraction
Tensor Tensor::operator- (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] - val_;
    }
    return result;
}

// Scalar Multiplication
Tensor Tensor::operator* (const double val_) const {
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] * val_;
    }
    return result;
}

// Scalar Division
Tensor Tensor::operator/ (const double val_) const {
    assert(val_ != 0.0);
    Tensor result(this -> rows(), this -> cols());
    for (size_t i = 0; i < this -> _data.size(); i++) {
        result._data[i] = this -> _data[i] / val_;
    }
    return result;
}

Tensor Tensor::operator+=(const double val_) {
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] += val_;
    }
    return *this;
}

Tensor Tensor::operator-=(const double val_) {
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] -= val_;
    }
    return *this;
}

Tensor Tensor::operator*=(const double val_) {
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] *= val_;
    }
    return *this;
}

Tensor Tensor::operator/=(const double val_) {
    assert(val_ != 0.0);
    for (size_t i = 0; i < this -> _data.size(); i++) {
        this -> _data[i] /= val_;
    }
    return *this;
}

axon::dtype::f64 frobenius_inner_product(const Tensor& a, const Tensor& b) {
    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("The shape must match for Frobenius Inner Product");
    }

    axon::dtype::f64 result = 0.0;
    for (size_t i = 0; i < a.get_size(); i++) {
        result += a(i) * b(i);
    }

    return result;
}


axon::dtype::f64 dot(const Tensor& a, const Tensor& b) {
    if (a.get_size() != b.get_size()) {
        throw std::invalid_argument("The number of elements must be same for both Tensors to perform a dot product");
    }
    axon::dtype::f64 result = 0.0;

    for (size_t i = 0; i < a.get_size(); i++) {
        result += a(i) * b(i);
    }

    return result;
}

Tensor operator+ (axon::dtype::f64 scalar, const Tensor& tensor) {
    return tensor + scalar;
}

Tensor operator* (axon::dtype::f64 scalar, const Tensor& tensor) {
    return tensor * scalar;
}

Tensor operator- (axon::dtype::f64 scalar, const Tensor& tensor) {
    Tensor result(tensor.rows(), tensor.cols());
    size_t t_size = tensor.get_size();

    for (size_t i = 0; i < t_size; i++) {
        result(i) = scalar - tensor(i);
    }

    return result;
}

Tensor operator/ (axon::dtype::f64 scalar, const Tensor& tensor) {
    Tensor result(tensor.rows(), tensor.cols());
    size_t t_size = tensor.get_size();
    
    for (size_t i = 0; i < t_size; i++) {
        // * NOTE: tensor(i) can be 0, which would result in `inf`
        result(i) = scalar / tensor(i);
    }

    return result;
}

bool operator== (const Tensor& a, const Tensor& b) {
    if (a.getShape() != b.getShape()) {
        throw std::invalid_argument("The shapes of the two tensors must match for this operation");
    }

    size_t size = a.get_size();
    for (size_t i = 0; i < size; i++) {
        if (std::abs(a(i) - b(i)) > axon::constants::eps) {
            return false;
        }         
    }
    return true;
}


// cache-friendly matmul
Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    size_t a_rows = a.rows();
    size_t a_cols = a.cols(); // shared dim
    size_t b_cols = b.cols();

    Tensor result = Tensor::zeros(a_rows, b_cols);

    // cache friendly loop
    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t k = 0; k < a_cols; ++k) {
            // Fetch a(i, k) once and reuse it across the inner loop
            double a_val = a(i, k); 
            for (size_t j = 0; j < b_cols; ++j) {
                result(i, j) += a_val * b(k, j);
            }
        }
    }

    result._prev = { const_cast<Tensor*>(&a), const_cast<Tensor*>(&b) };

    result._backward_fn = [pa = &a, pb = &b](Tensor* self) mutable {
        // Gradient for A = current @ b.T
        *(pa -> _grad) += matmul(*(self -> _grad), pb -> T());
    
        // Gradient for B = a.T @ current
        *(pb -> _grad) += matmul(pa -> T(), *(self -> _grad));
    };

    return result;
}

// Transpose
Tensor Tensor::T() const {
    Tensor result(cols(), rows());
    for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < cols(); j++) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

