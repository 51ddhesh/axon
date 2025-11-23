#ifndef AXON_TENSOR_HPP
#define AXON_TENSOR_HPP

#include <vector>
#include <memory>
#include <functional>
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <initializer_list>
#include <cassert>

namespace axon {

    class Tensor;

    struct TensorImpl {
        std::vector<double> data;
        std::vector<double> grad;
        std::vector<size_t> shape;
        std::vector<size_t> strides;

        std::vector<Tensor> prev; 
        std::function<void(Tensor&)> backward_fn;

        TensorImpl(const std::vector<size_t>& shape, const std::vector<double>& data);
    };

    class Tensor {
    private:
        std::shared_ptr<TensorImpl> impl;

    public:
        Tensor();
        Tensor(double val);
        Tensor(std::vector<size_t> shape, bool requires_grad = false);
        Tensor(std::initializer_list<double> list);
        static Tensor from_2d(std::initializer_list<std::initializer_list<double>> list);
        
        double& item(size_t i);
        double item(size_t i) const;
        const std::vector<size_t>& shape() const;
        size_t size() const;
        
        // CHANGED: Added 'const' to these two methods
        double* data_ptr() const;
        double* grad_ptr() const;

        void set_backward_fn(std::vector<Tensor> prev, std::function<void(Tensor&)> fn);
        void backward();
        void zero_grad();

        friend Tensor operator+(const Tensor& a, const Tensor& b);
        friend Tensor operator-(const Tensor& a, const Tensor& b);
        friend Tensor operator*(const Tensor& a, const Tensor& b);
        friend Tensor matmul(const Tensor& a, const Tensor& b);
        
        static Tensor relu(const Tensor& input);
        static Tensor pow(const Tensor& base, double exp);
        Tensor sum() const;
        
        static Tensor create_from_impl(std::shared_ptr<TensorImpl> impl);
    };

    void print(const Tensor& t);

} // namespace axon

#endif // AXON_TENSOR_HPP