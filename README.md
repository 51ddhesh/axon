# Axon
- Branch `master`

[**`AXON`**](https://github.com/51ddhesh/axon/) is a simple deep learning library in the process inspired from `PyTorch`.

---

## Project Structure

```
axon/
 ├── README.md
 ├── CMakeLists.txt
 ├── LICENSE
 ├── main.cpp
 ├── include/
 │   ├── Activations.hpp
 │   ├── Linear.hpp
 │   ├── LossFunctions.hpp
 │   ├── Sequential.hpp
 │   ├── Tensor.hpp
 │   ├── TensorMath.hpp
 │   └── private/
 │       └── OperationHelpers.hpp
 ├── src/
 │   ├── core/
 │   │   ├── tensor_core.cpp
 │   │   ├── tensor_ops.cpp
 │   │   └── tensor_utils.cpp
 │   ├── math/
 │   │   ├── binary_ops.cpp
 │   │   ├── comparisons.cpp
 │   │   └── unary_ops.cpp
 │   └── nn/
 │       ├── activations.cpp
 │       ├── linear.cpp
 │       ├── loss.cpp
 │       ├── sequential.cpp
 │       └── autograd/
 │           └── backward.cpp
 └── utils/
     └── random_.hpp
```

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
