# Axon
- Branch `tensor/double`

[**`AXON`**](https://github.com/51ddhesh/axon/) is a simple header-only deep learning library in the process inspired from `PyTorch`.


--- 

### Prerequisites
1. A C++ Compiler 
2. CMake


### Usage:
1. Clone the project:

```bash
git clone https://github.com/51ddhesh/axon/ && cd axon
```

2. Create the `build` directory

```
mkdir build
```

3. Create the build files

```
cmake -S . -B build/
```

4. Compile 

```
cmake --build build
```
The creates an executable `ax` in `build/`

5. Run the executable

```bash
./build/ax
```



## TODOs:
- [ ] Implement softmax in `src/nn/`
- [ ] Implement Transpose in `src/core/tensor_ops.cpp`


