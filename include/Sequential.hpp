// include/Sequential.hpp
// github.com/51ddhesh/axon/
// MIT License

#ifndef AXON_SEQUENTIAL_HPP
#define AXON_SEQUENTIAL_HPP

#include "Linear.hpp"

class Sequential {
private:
    std::vector<Linear> _layers;

public:
    // Default Constructor
    Sequential() {}

    // Add -> add layers to the network
    void add(const Linear& layer);

    // Forward pass -> `torch` style
    Tensor linear(const Tensor& input) const;

    // Forward pass -> `tf` style
    Tensor forward(const Tensor& input) const;
};

#endif // AXON_SEQUENTIAL_HPP