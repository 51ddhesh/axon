# Axon

A C++ deep learning framework with CUDA support, automatic differentiation, and neural network modules.

## Version

v3.1

## Features

- Tensor operations with broadcasting support
- Automatic differentiation (autograd)
- GPU acceleration via CUDA
- Neural network modules (Linear, Embedding, LayerNorm, MultiHeadAttention, GPT-2)
- Optimizers (SGD, AdamW)
- Model serialization (save/load)

## Requirements

- C++20 compatible compiler
- CUDA Toolkit (optional, for GPU support)
- CMake 3.20+

## Building

```bash
mkdir build && cd build
cmake ..
make -j4
```

## Quick Example

```cpp
#include "axon/tensor.hpp"
#include "axon/ops.hpp"
#include "axon/optimizer.hpp"

using namespace axon;

int main() {
    auto x = Tensor::zeros({4, 2});
    x.at({0,0}) = 0; x.at({0,1}) = 0;
    x.at({1,0}) = 0; x.at({1,1}) = 1;
    x.at({2,0}) = 1; x.at({2,1}) = 0;
    x.at({3,0}) = 1; x.at({3,1}) = 1;

    auto y = Tensor::zeros({4, 1});
    y.at({0,0}) = 0; y.at({1,0}) = 1;
    y.at({2,0}) = 1; y.at({3,0}) = 0;

    auto w1 = Tensor::zeros({2, 8});
    auto b1 = Tensor::zeros({1, 8});
    auto w2 = Tensor::zeros({8, 1});
    auto b2 = Tensor::zeros({1, 1});

    for(size_t i=0; i<w1.numel(); i++)
        w1.data_ptr()[i] = (((float)rand()/RAND_MAX) - 0.5f) * 2.0f;
    for(size_t i=0; i<w2.numel(); i++)
        w2.data_ptr()[i] = (((float)rand()/RAND_MAX) - 0.5f) * 2.0f;

    w1.set_requires_grad(true); b1.set_requires_grad(true);
    w2.set_requires_grad(true); b2.set_requires_grad(true);

    SGD optimizer({w1, b1, w2, b2}, 0.05f);

    for (int epoch = 0; epoch < 2000; ++epoch) {
        auto h1 = relu(matmul(x, w1) + b1);
        auto out = matmul(h1, w2) + b2;
        auto loss = sum((out - y) * (out - y));

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (epoch % 200 == 0)
            std::cout << "Epoch " << epoch << " | Loss: " << loss.at({0}) << "\n";
    }

    return 0;
}
```

## Legacy Versions

- v1, v2: [github.com/51ddhesh/axon-legacy](https://github.com/51ddhesh/axon-legacy)
- v3: [github.com/51ddhesh/axon3](https://github.com/51ddhesh/axon3)

## License

MIT License. See [LICENSE](./LICENSE) for details.