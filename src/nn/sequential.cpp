// src/nn/sequential.cpp
// github.com/51ddhesh/axon
// MIT License

#include "../../include/Sequential.hpp"

void Sequential::add(const Linear& layer) {
    _layers.push_back(layer);
}

Tensor Sequential::linear(const Tensor& input) const {
    Tensor current = input;
    for (const auto& layer : _layers) {
        current = layer.linear(current);
    }
    return current;
}

Tensor Sequential::forward(const Tensor& input) const {
    return this -> linear(input);
}
