// include/Tensor.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <iostream>
#include <vector>
#include <initializer_list>

namespace axon {
    using i64 = long long;
}



template <typename T = double>
class Tensor {
private:
    std::vector<T> data;
    
public:
    Tensor(std::initializer_list<T> init_list) : data(init_list) {}

    Tensor(const std::vector<T>& vec) : data(std::move(vec)) {}
    
    Tensor() {}

    const std::vector<T>& getData() const {
        return this->data;
    }

    void print_tensor() {
        auto data = getData();
        std::cout << "Tensor([";
        for (auto i : data) {
            std::cout << i << ',';
        }
        std::cout << "])" << std::endl;
    }

    Tensor operator+(const Tensor& other) const {
        if (getData().size() != other.getData().size()) {
            throw std::range_error("Length of the Tensors must be same");
        }
        Tensor result(std::vector<T> (other.data.size(), 0));
        for (size_t i = 0; i < other.data.size(); i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (getData().size() != other.getData().size()) {
            throw std::range_error("Length of the Tensors must be same");
        }
        Tensor result(std::vector<T> (other.data.size(), 0));
        for (size_t i = 0; i < other.data.size(); i++) {
            result.data[i] = data[i] - other.data[i];
        }
        return result;
    }
    
    Tensor operator*(const Tensor& other) const {
        if (getData().size() != other.getData().size()) {
            throw std::range_error("Length of the Tensors must be same");
        }
        Tensor result(std::vector<T> (other.data.size(), 0));
        for (size_t i = 0; i < other.data.size(); i++) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }
};

template <typename T>
void print(Tensor<T>& t) {
    auto data = t.getData();
    std::cout << "Tensor([";
    for (auto i : data) {
        std::cout << i << ',';
    }
    std::cout << "])" << std::endl;
}

template <typename T, typename U>
axon::i64 dot(const Tensor<T>& a, const Tensor<U>& b) {
    if (a.getData().size() != b.getData().size()) {
        throw std::range_error("The size of both Tensors must be the same");
    }
    axon::i64 dot = 0;
    size_t n = a.getData().size();
    
    for (size_t i = 0; i < n; i++) {
        dot += int(double(a.getData()[i]) * double(b.getData()[i]));
    }
    
    return dot;
}


#endif // AXON_TENSOR_HPP