// include/tensor.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <iostream>
#include <vector>
#include <initializer_list>
#include <iomanip>

namespace axon {
    using i64 = long long;
    using f64 = double;
}

class Tensor {
private:
    std::vector<double> data;
    
public:
    Tensor(std::initializer_list<double> init_list) : data(init_list) {}

    Tensor(const std::vector<double>& vec) : data(std::move(vec)) {}
    
    Tensor() {}

    const inline std::vector<double>& getData() const {
        return this->data;
    }

        inline size_t getSize() const {
        return this->data.size();
    }

    inline Tensor zeros(size_t size_) {
        std::vector<double> t(size_, 0);
        return Tensor(t);
    }

    void print_tensor();

    // Operations with other Tensors
    Tensor operator+(const Tensor& other) const;

    Tensor operator-(const Tensor& other) const;
    
    Tensor operator*(const Tensor& other) const;

    // Operations with scalars
    Tensor operator+(const double val_) const;
    
    Tensor operator-(const double val_) const;
    
    Tensor operator*(const double val_) const;
};

void print(const Tensor& t);

axon::f64 dot(const Tensor& a, const Tensor& b); 

#endif // AXON_TENSOR_HPP