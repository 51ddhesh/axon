# Axon

A high-performance C++ deep learning framework.

**Status:** v3.2 - v4.0 under development (see `v4` branch)

---

## About

Axon is a deep learning framework written in C++20. Features:

- **Tensor operations** with automatic differentiation
- **Neural network layers** (Linear, ReLU, Embedding, LayerNorm, Attention)
- **Optimizers** (SGD, Adam, AdamW)
- **CPU and CUDA support**

---
## Building

### Requirements
- CMake 3.20+
- C++20 compiler
- CUDA Toolkit (optional, for GPU support)

### Build
```bash
mkdir build && cd build
cmake .. -DCUDA_ENABLED=ON  # or OFF for CPU-only
make
```

### Run Examples
```bash
# With CUDA
g++ -std=c++20 -I include examples/1_Basic_Ops.cpp -L build -laxon -lcublas -lcudart -o ex1
./ex1

# CPU-only (requires fixing CUDA guards in source)
g++ -std=c++20 -I include examples/1_Basic_Ops.cpp -L build -laxon -o ex1
./ex1
```

---

## License

Licensed under the MIT License. Check [LICENSE](./LICENSE) for details. 
