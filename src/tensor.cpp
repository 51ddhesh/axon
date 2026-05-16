#include "axon/tensor.hpp"
#include "axon/kernels.hpp"
#include "axon/autograd.hpp"
#include "axon/grad_mode.hpp"
#include "axon/ops.hpp"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

namespace axon {

    Tensor::Tensor(std::vector<int> shape, Device dev)
        : shape(shape), offset(0) {
        calculate_strides();
        storage = std::make_shared<Storage>(size * sizeof(float), dev);
        state = std::make_shared<TensorState>();
    }

    Tensor::Tensor(const Tensor& other)
        : storage(other.storage),
          state(other.state),
          shape(other.shape),
          stride(other.stride),
          offset(other.offset),
          size(other.size) {}

    Tensor::Tensor(Tensor&& other) noexcept
        : storage(std::move(other.storage)),
          state(std::move(other.state)),
          shape(std::move(other.shape)),
          stride(std::move(other.stride)),
          offset(other.offset),
          size(other.size) {
        other.offset = 0;
        other.size = 0;
    }

    Tensor& Tensor::operator=(const Tensor& other) {
        if (this != &other) {
            storage = other.storage;
            state = other.state;
            shape = other.shape;
            stride = other.stride;
            offset = other.offset;
            size = other.size;
        }
        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            storage = std::move(other.storage);
            state = std::move(other.state);
            shape = std::move(other.shape);
            stride = std::move(other.stride);
            offset = other.offset;
            size = other.size;
            other.offset = 0;
            other.size = 0;
        }
        return *this;
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

    Tensor Tensor::zeros(std::vector<int> shape, Device dev) {
        Tensor t(shape, dev);
        if (t.size > 0 && t.device().type == DeviceType::CPU) {
            std::memset(t.data_ptr(), 0, t.size * sizeof(float));
        } else {
            t.storage -> allocator -> set_zero(t.data_ptr(), t.size * sizeof(float)); 
        }
        return t;
    }

    Tensor Tensor::ones(std::vector<int> shape, Device dev) {
        Tensor t(shape, dev);
        if (t.size > 0 && t.device().type == DeviceType::CPU) {
            kernels::cpu::fill_f32(t.numel(), 1.0f, t.data_ptr());
        } else {
            kernels::gpu::fill_f32(t.numel(), 1.0f, t.data_ptr());
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

    Tensor Tensor::to(Device target_device) const {
        if (device() == target_device) return *this;

        // Ensure contiguous 
        Tensor src = this -> is_contiguous() ? *this : this -> contiguous();
        // Allocate memory on target device
        Tensor dst = Tensor::zeros(src.get_shape(), target_device);

        // Copy the data

        // 1. CPU -> GPU
        if (src.device().type == DeviceType::CPU && target_device.type == DeviceType::CUDA) {
            cudaMemcpy(dst.data_ptr(), src.data_ptr(), src.numel() * sizeof(float), axon::MemcpyHostToDevice);
        }
        // 2. GPU -> CPU
        else if (src.device().type == DeviceType::CUDA && target_device.type == DeviceType::CPU) {
            cudaMemcpy(dst.data_ptr(), src.data_ptr(), src.numel() * sizeof(float), axon::MemcpyDeviceToHost);
        }

        else {
            throw std::runtime_error("Multiple GPUs/CPUs not supported yet!!");
        }

        return dst;
    }

    void Tensor::zero_grad() {
        set_grad(nullptr);
    }

    void Tensor::add_grad(const Tensor& new_grad) {
        auto g = get_grad();
        if (!g) {
            set_grad(std::make_shared<Tensor>(new_grad.contiguous()));
        } else {
            *g = axon::add(*g, new_grad);
        }
    }

    void build_topo(Tensor* t,
                    std::vector<GradFn*>& topo,
                    std::unordered_set<GradFn*>& visited,
                    std::unordered_map<GradFn*, Tensor*>& fn_to_tensor_map) {

        auto fn = t->get_grad_fn().get();
        if (!fn || visited.count(fn)) return;

        visited.insert(fn);
        fn_to_tensor_map[fn] = t;

        for (auto& edge : fn->next_edges) {
            if (edge.input_tensor) {
                build_topo(edge.input_tensor.get(), topo, visited, fn_to_tensor_map);
            }
        }

        topo.push_back(fn);
    }

    void Tensor::backward() {
        auto _grad = get_grad();
        if (!_grad) {
            if (numel() != 1) {
                throw std::runtime_error("[BACKWARD] Error: grad can only be created for scalar outputs. Use grad for non-scalar.");
            }
            _grad = std::make_shared<Tensor>(Tensor::ones(shape));
            set_grad(_grad);
        }

        std::vector<GradFn*> topo_order;
        std::unordered_set<GradFn*> visited;
        std::unordered_map<GradFn*, Tensor*> fn_to_tensor_map;

        build_topo(this, topo_order, visited, fn_to_tensor_map);

        std::reverse(topo_order.begin(), topo_order.end());

        for (GradFn* fn : topo_order) {
            Tensor* output_tensor = fn_to_tensor_map[fn];
            auto output_grad = output_tensor->get_grad();

            if (!output_grad) continue;

            auto input_grads = fn->apply(*output_grad);

            if (input_grads.size() != fn->next_edges.size()) {
                std::cerr << "[FATAL] Autograd Graph Mismatch.\n";
                std::terminate();
            }

            for (size_t i = 0; i < fn->next_edges.size(); i++) {
                auto& edge = fn->next_edges[i];
                if (edge.input_tensor && edge.input_tensor->requires_grad()) {
                    edge.input_tensor->add_grad(input_grads[i]);
                }
            }
        }
    }

}