// src/tensor_utils.cpp
// github.com/51ddhesh
// MIT License

#include "../../include/Tensor.hpp"
#include "../../utils/random_.hpp"
#include <iomanip>

void Tensor::print_tensor() const {
    std::cout << "Tensor([";
    for (size_t i = 0; i < rows(); i++) {
        if (i > 0) {
            std::cout << ' ';
        }
        if (i != 0) std::cout << "       ";
        std::cout << '[';
        for (size_t j = 0; j < cols(); j++) {
            std::cout << std::fixed << std::setprecision(4) << (*this)(i, j);
            if (j < cols() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ']';
        if (i < rows() - 1) {
            std::cout << ",\n";
        }
    }
    std::cout << "])" << std::endl;
}

// Wrapper over `Tensor::print_tensor
void print(const Tensor& t) {
    t.print_tensor();
}

Tensor Tensor::zeros(size_t rows_, size_t cols_) {
    return Tensor(rows_, cols_);
}

Tensor Tensor::ones(size_t rows_, size_t cols_) {
    return Tensor(rows_, cols_, 1.0);
}

Tensor Tensor::randint(size_t rows_, size_t cols_) {
    int min_ = -1000;
    int max_ = 1000;
    Tensor random_int_tensor(rows_, cols_);
    for (size_t i = 0; i < random_int_tensor._data.size(); i++) {
        random_int_tensor(i) = axon_random::random_int(min_, max_);
    }
    return random_int_tensor;
}

Tensor Tensor::randint(size_t rows_, size_t cols_, std::vector<axon::dtype::i32> limits_) {
    assert(limits_.size() == 2);
    Tensor random_int_tensor(rows_, cols_);
    for (size_t i = 0; i < random_int_tensor._data.size(); i++) {
        random_int_tensor(i) = axon_random::random_int(limits_[0], limits_[1]);
    }
    return random_int_tensor;
}

Tensor Tensor::randn(size_t rows_, size_t cols_, std::vector<axon::dtype::f64> limits_) {
    assert(limits_.size() == 2);
    Tensor random_tensor(rows_, cols_);
    for (size_t i = 0; i < random_tensor.get_size(); i++) {
        random_tensor(i) = axon_random::random_double(limits_[0], limits_[1]);
    }
    return random_tensor;
}

Tensor Tensor::randn(size_t rows_, size_t cols_) {
    axon::dtype::f64 min_ = 0.0, max_ = 1.0;
    Tensor random_tensor(rows_, cols_);
    for (size_t i = 0; i < random_tensor.get_size(); i++) {
        random_tensor(i) = axon_random::random_double(min_, max_);
    }
    return random_tensor;
}

Tensor Tensor::row(std::initializer_list<axon::dtype::f64> init_list) {
    return Tensor(init_list);
}

Tensor Tensor::column(std::initializer_list<axon::dtype::f64> init_list) {
    Tensor result(init_list.size(), 1);
    size_t i = 0;
    for (const axon::dtype::f64& val : init_list) {
        result(i, 0) = val;
        i++;
    }
    return result;
}

Tensor Tensor::sum() const {
    axon::dtype::f64 _sum = 0;
    size_t tensor_size = this -> get_size();
    for (size_t i = 0; i < tensor_size; i++) {
        _sum += (*this)(i);
    }

    Tensor result(1, 1, _sum);

    result._prev = { const_cast<Tensor*>(this) };
    result._backward_fn = [this_ptr = const_cast<Tensor*>(this)](Tensor* self) {
        axon::dtype::f64 grad_val = (*(self -> _grad))(0, 0);
        Tensor grad_update = Tensor::ones(this_ptr -> rows(), this_ptr -> cols()) * grad_val;
        *(this_ptr -> _grad) += grad_update;
    };
    return result;
}

Tensor Tensor::sum(int axis) const {
    if (axis == 0) {
        Tensor result(1, this -> cols());
        for (size_t i = 0; i < this -> cols(); i++) {
            axon::dtype::f64 col_sum = 0.0;
            for (size_t j = 0; j < this -> rows(); j++) {
                col_sum += (*this)(j, i);
            }
            result(0, i) = col_sum;
        }
        return result;
    } else if (axis == 1) {
        Tensor result(this -> rows(), 1);
        for (size_t i = 0; i < this -> rows(); i++) {
            axon::dtype::f64 row_sum = 0.0;
            for (size_t j = 0; j < this -> cols(); j++) {
                row_sum += (*this)(i, j);
            }
            result(i, 0) = row_sum;
        }
        return result;
    }
    throw std::invalid_argument("Axis must be 0 or 1");
}

void Tensor::zero_grad() {
    if (_grad) {
        std::fill(_grad -> _data.begin(), _grad -> _data.end(), 0.0);
    }
}
