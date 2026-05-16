#pragma once

#include "storage.hpp"
#include <vector>
#include <memory>
#include <string>

namespace axon {

    struct GradFn;
    class Tensor;

    struct TensorState {
        bool requires_grad = false;
        std::shared_ptr<Tensor> grad = nullptr;
        std::shared_ptr<GradFn> grad_fn = nullptr;
    };

    class Tensor {
    private:
        std::shared_ptr<Storage> storage;
        std::shared_ptr<TensorState> state;
        std::vector<int> shape;
        std::vector<int> stride;
        int offset;
        size_t size;

        void calculate_strides();

    public:
        Tensor(std::vector<int> shape, Device dev = Device(DeviceType::CPU));

        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(const Tensor& other);
        Tensor& operator=(Tensor&& other) noexcept;
        ~Tensor() = default;

        static Tensor zeros(std::vector<int> shape, Device dev = Device(DeviceType::CPU));
        static Tensor ones(std::vector<int> shape, Device dev = Device(DeviceType::CPU));

        static Tensor from_storage(std::shared_ptr<Storage> storage,
            std::vector<int> shape, std::vector<int> stride,
            int offset
        );

        [[nodiscard]] const std::vector<int>& get_shape() const {
            return shape;
        }
        
        [[nodiscard]] const std::vector<int>& get_stride() const {
            return stride;
        }
        
        [[nodiscard]] int get_offset() const {
            return offset;
        }
        
        [[nodiscard]] size_t numel() const {
            return size;
        }

        [[nodiscard]] std::shared_ptr<Storage> get_storage() const {
            return storage;
        }

        Device device() const {
            return storage -> device;
        }

        float* data_ptr() {
            return storage -> ptr<float>() + offset;
        }
        
        const float* data_ptr() const {
            return storage -> ptr<float>() + offset;
        }

        bool requires_grad() const {
            return state -> requires_grad;
        }

        void set_requires_grad(bool b) {
            state -> requires_grad = b;
        }

        
        std::shared_ptr<Tensor> get_grad() const {
            return state -> grad;
        }

        void set_grad(std::shared_ptr<Tensor> g) {
            state -> grad = g;
        }


        std::shared_ptr<GradFn> get_grad_fn() const {
            return state -> grad_fn;
        }

        void set_grad_fn(std::shared_ptr<GradFn> fn) {
            state -> grad_fn = fn;
        }


        void backward();
        void add_grad(const Tensor& new_grad);
        void zero_grad();
        bool is_leaf() const { 
            return state -> grad_fn == nullptr; 
        }


        // Utils
        bool is_contiguous() const;
        void print() const;
        float& at(const std::vector<int>& indices);
    
        Tensor expand(const std::vector<int>& target_shape) const;

        Tensor contiguous() const;

        
        Tensor to(Device target_device) const;
    };

    void save_model(const std::vector<Tensor>& params, const std::string& filepath);
    void load_model(std::vector<Tensor>& params, const std::string& filepath);
} // namespace axon
