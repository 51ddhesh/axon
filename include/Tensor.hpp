// include/Tensor.hpp
// github.com/51ddhesh
// MIT License

#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <iostream>
#include <vector>

template <typename T>
class Tensor {
private:
    std::vector<T> data;

public:
    Tensor(std::vector<T>& initializer_list) {
        data = initializer_list;
    }

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