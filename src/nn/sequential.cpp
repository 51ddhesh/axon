// src/nn/sequential.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/Sequential.hpp"

void Sequential::add(const Linear& layer) {
    _layers.push_back(layer);
}

Tensor Sequential::linear(const Tensor& input) {
    Tensor current = input;
    for (auto& layer : _layers) {
        current = layer.linear(current);
    }
    return current;
}

Tensor Sequential::forward(const Tensor& input) {
    return this -> linear(input);
}

std::vector<Tensor*> Sequential::parameters() {
    std::vector<Tensor*> params;
    for (auto& layer : _layers) {
        auto layer_params = layer.parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}
