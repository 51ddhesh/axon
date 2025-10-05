// include/Tensor.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <iostream>
#include <vector>
#include <initializer_list>

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

#endif // AXON_TENSOR_HPP