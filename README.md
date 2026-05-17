# Axon

A high-performance C++ deep learning framework.

**Version:** v4.0.0 (under development)

---

## Status

Work in progress on v4 architecture with modular design.

---

## Building

### Requirements

- CMake 3.20+
- C++20 compiler
- CUDA Toolkit (optional, for GPU support)

### Build

```bash
# Clone and setup
mkdir build && cd build

# Configure (CUDA enabled by default)
cmake ..

# Or without CUDA
cmake .. -DCUDA_ENABLED=OFF

# Build
make -j$(nproc)
```

### Run Tests

```bash
./tests/axon_test
# Or with ctest
ctest --output-on-failure
```

---

## Project Structure

```
axon/
├── include/axon/      # Headers
│   └── core/          # Device, Storage, Arena
├── src/               # Implementation
└── tests/             # Unit tests
    └── core/          # Core layer tests
```

---

## License

Boost Software License - Version 1.0. See [LICENSE](LICENSE) file.

---

## Legacy Versions

- v3.2: [github.com/51ddhesh/axon3](https://github.com/51ddhesh/axon3)
- v1, v2: [github.com/51ddhesh/axon-legacy](https://github.com/51ddhesh/axon-legacy)