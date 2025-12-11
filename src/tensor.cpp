#include "axon/tensor.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstring> 
#include <stdexcept>

namespace axon {

    Tensor::Tensor(std::vector<int> shape) 
        : shape(shape), offset(0) {
        calculate_strides();
        storage = std::make_shared<Storage>(size * sizeof(float));
        state = std::make_shared<TensorState>();
    }

    Tensor Tensor::from_storage(
        std::shared_ptr<Storage> storage, 
        std::vector<int> shape, std::vector<int> stride, 
        int offset) {
        
        Tensor t({1}); 
        
        t.storage = storage;
        t.shape = shape;
        t.stride = stride;
        t.offset = offset;
        t.state = std::make_shared<TensorState>();
        
        t.size = 1;
        for(int s : shape) t.size *= s;
        
        return t;
    }

    void Tensor::calculate_strides() {
        stride.resize(shape.size());
        size_t running_size = 1;
        
        if (!shape.empty()) {
            for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
                stride[i] = running_size;
                running_size *= shape[i];
            }
        }
        
        this -> size = running_size;
    }

    Tensor Tensor::zeros(std::vector<int> shape) {
        Tensor t(shape);
        if (t.size > 0 && t.device().type == DeviceType::CPU) {
            std::memset(t.data_ptr(), 0, t.size * sizeof(float));
        } else {
            // TODO: call a CUDA kernel 
        }
        return t;
    }

    Tensor Tensor::ones(std::vector<int> shape) {
        Tensor t(shape);
        if (t.size > 0 && t.device().type == DeviceType::CPU) {
            std::fill(t.data_ptr(), t.data_ptr() + t.size, 1.0f);
        } else {
            // TODO: call a CUDA kernel
        }
        return t;
    }

    bool Tensor::is_contiguous() const {
        int z = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            if (shape[i] != 1) {
                if (stride[i] != z) return false;
                z *= shape[i];
            }
        }
        return true;
    }

    float& Tensor::at(const std::vector<int>& indices) {
        if (indices.size() != shape.size()) {
            throw std::invalid_argument("[AT]: Dim mismatch");
        }

        int flat = offset;
        for (size_t i = 0; i < indices.size(); i++) {
            flat += stride[i] * indices[i];
        }
        
        if (flat < 0 || (flat * sizeof(float))>= storage -> nbytes) {
            throw std::out_of_range("[AT] Error: Tensor::at() computed offset outside Storage bounds");
        }

        return storage -> ptr<float>()[flat];
    }

    void print_recursive(const Tensor& t, int dim, int current_offset, std::vector<int>& indices) {
        std::string indent(dim, ' ');
        int dim_len = t.get_shape()[dim];
        int dim_stride = t.get_stride()[dim];

        std::cout << indent << "[";
        if (dim == t.get_shape().size() - 1) {
            // Base: Print row
            for (int i = 0; i < dim_len; ++i) {
                std::cout << t.data_ptr()[current_offset + i * dim_stride]; 
                if (i < dim_len - 1) std::cout << ", ";
            }
        } else {
            // Recursive
            std::cout << "\n";
            for (int i = 0; i < dim_len; ++i) {
                indices.push_back(i);
                print_recursive(t, dim + 1, current_offset + i * dim_stride, indices);
                indices.pop_back();
                if (i < dim_len - 1) std::cout << ",\n";
            }
            std::cout << "\n" << indent;
        }
        std::cout << "]";
    }

    void Tensor::print() const {
        std::cout << "Tensor Shape={";
        for(auto s : shape) std::cout << s << ",";
        std::cout << "} Stride={";
        for(auto s : stride) std::cout << s << ",";
        std::cout << "}\n";

        if (numel() > 0) {
            std::vector<int> indices;
            print_recursive(*this, 0, 0, indices); 
            std::cout << "\n";
        }
        std::cout << "---------------------------\n";
    }

    Tensor Tensor::expand(const std::vector<int>& target_shape) const {
        if (shape.size() > target_shape.size()) {
            throw std::invalid_argument("[EXPAND] Error: Cannot broadcast to a smaller rank");
        }

        std::vector<int> aligned_shape(target_shape.size(), 1);
        std::vector<int> aligned_stride(target_shape.size(), 0);
    
        int offset_dim = static_cast<int>(target_shape.size() - shape.size());
        
        size_t shape_size = shape.size();
        size_t target_shape_size = target_shape.size();

        for (size_t i = 0; i < shape_size; i++) {
            aligned_shape[offset_dim + i] = shape[i];
            aligned_stride[offset_dim + i] = stride[i];
        }
    
        std::vector<int> new_strides(target_shape_size);

        for (size_t i = 0; i < target_shape_size; i++) {
            int target_dim = target_shape[i];
            int current_dim = aligned_shape[i];
            int current_stride = aligned_stride[i];

            if (target_dim == current_dim) {
                new_strides[i] = current_stride;
            } else if (current_dim == 1) {
                new_strides[i] = 0;
            } else {
                throw std::invalid_argument("[EXPAND] Error: Cannot broadcast");
            }
        }

        return Tensor::from_storage(storage, target_shape, new_strides, offset);
    }

    void copy_recursive(
        int dim, const Tensor& src, int src_offset,
        float* dst_ptr, int& dst_index) {
        int dim_len = src.get_shape()[dim];
        int dim_stride = src.get_stride()[dim];

        if (dim == src.get_shape().size() - 1) {
            // Base Case
            for (int i = 0; i < dim_len; i++) {
                dst_ptr[dst_index++] = src.data_ptr()[src_offset + i * dim_stride];
            }
        } else {
            for (int i = 0; i < dim_len; i++) {
                copy_recursive(dim + 1, src, src_offset + i * dim_stride, dst_ptr, dst_index);
            }
        }
    }

    Tensor Tensor::contiguous() const {
        if (is_contiguous()) {
            // Deep copy
            Tensor out = Tensor::zeros(shape); // Allocates new storage
            if (device().type == DeviceType::CPU) {
                std::memcpy(out.data_ptr(), this->data_ptr(), size * sizeof(float));
            } 
            return out;
        } else {
            Tensor out = Tensor::zeros(shape);
            int dst_index = 0;
            copy_recursive(0, *this, 0, out.data_ptr(), dst_index);
            return out;
        }
    }
}