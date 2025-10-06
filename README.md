# Axon
- Branch `tensor/double`

[**`AXON`**](https://github.com/51ddhesh/axon/) is a simple deep learning library in the process inspired from `PyTorch`.


--- 

### Current Features:
- Supports a `1D` `tensor` like `torch` with addition, subtraction and multiplication operation (overloaded `+`, `-`, `*` operators).
- Supports operations between a `tensor` and scalars.
- Super Basic `dot product` between two `vectors`.

> See sample usage in [main.cpp](./main.cpp)

---

**Current Idea**: Port the `Tensor` class to a `double` rather than a templated class. 


### TODOs:
- [x] Port the [`Tensor` class](./include/tensor.hpp) to a `double`.
- [x] Add more datatypes in the `axon` namespace (added `axon::i64` and `axon::f64`).


