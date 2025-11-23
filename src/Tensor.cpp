#include "../include/Tensor.hpp"
#include <random>
#include <unordered_set>

namespace axon {

    // --- TensorImpl Implementation ---
    TensorImpl::TensorImpl(const std::vector<size_t>& s, const std::vector<double>& d) 
        : data(d), shape(s) {
        grad.resize(data.size(), 0.0);
        
        strides.resize(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // --- Tensor Implementation ---

    Tensor::Tensor() {}

    Tensor Tensor::create_from_impl(std::shared_ptr<TensorImpl> impl) {
        Tensor t;
        t.impl = impl;
        return t;
    }

    Tensor::Tensor(double val) {
        impl = std::make_shared<TensorImpl>(std::vector<size_t>{1}, std::vector<double>{val});
    }

    Tensor::Tensor(std::vector<size_t> shape, bool random) {
        size_t size = 1;
        for (auto s : shape) size *= s;
        std::vector<double> d(size);
        
        if (random) {
            static std::mt19937 gen(42);
            std::uniform_real_distribution<> dis(-1.0, 1.0); 
            for (auto& v : d) v = dis(gen);
        }
        impl = std::make_shared<TensorImpl>(shape, d);
    }

    Tensor::Tensor(std::initializer_list<double> list) {
        impl = std::make_shared<TensorImpl>(
            std::vector<size_t>{list.size()}, 
            std::vector<double>(list)
        );
    }

    Tensor Tensor::from_2d(std::initializer_list<std::initializer_list<double>> list) {
        size_t rows = list.size();
        size_t cols = list.begin()->size();
        std::vector<double> flat;
        for (auto& row : list) flat.insert(flat.end(), row.begin(), row.end());
        
        Tensor t;
        t.impl = std::make_shared<TensorImpl>(std::vector<size_t>{rows, cols}, flat);
        return t;
    }

    double& Tensor::item(size_t i) { return impl->data[i]; }
    double Tensor::item(size_t i) const { return impl->data[i]; }
    const std::vector<size_t>& Tensor::shape() const { return impl->shape; }
    size_t Tensor::size() const { return impl->data.size(); }
    
    // CHANGED: Updated to const
    double* Tensor::data_ptr() const { return impl->data.data(); }
    double* Tensor::grad_ptr() const { return impl->grad.data(); }

    void Tensor::zero_grad() {
        if(impl) std::fill(impl->grad.begin(), impl->grad.end(), 0.0);
    }

    void Tensor::set_backward_fn(std::vector<Tensor> prev, std::function<void(Tensor&)> fn) {
        impl->prev = prev;
        impl->backward_fn = fn;
    }

    // --- Autograd Engine ---

    void Tensor::backward() {
        std::vector<Tensor> sorted;
        std::unordered_set<TensorImpl*> visited;

        std::function<void(Tensor&)> build = [&](Tensor& node) {
            if (visited.find(node.impl.get()) != visited.end()) return;
            visited.insert(node.impl.get());

            for (auto& parent : node.impl->prev) {
                build(parent);
            }
            sorted.push_back(node);
        };

        build(*this);
        std::reverse(sorted.begin(), sorted.end());

        std::fill(impl->grad.begin(), impl->grad.end(), 1.0);

        for (auto& node : sorted) {
            if (node.impl->backward_fn) {
                node.impl->backward_fn(node);
            }
        }
    }

    // --- Operators ---

    Tensor operator+(const Tensor& a, const Tensor& b) {
        bool broadcast_b = (b.shape().size() > 0 && b.shape()[0] == 1 && a.shape().size() > 1 && b.shape()[1] == a.shape()[1]);
        
        Tensor res(a.shape(), false);
        size_t N = (a.shape().size() > 1) ? a.shape()[1] : 1;

        for(size_t i=0; i<a.size(); ++i) {
            double b_val = broadcast_b ? b.item(i % N) : b.item(i);
            res.item(i) = a.item(i) + b_val;
        }

        res.set_backward_fn({a, b}, [a, b, broadcast_b, N](Tensor& self) mutable {
            for(size_t i=0; i<self.size(); ++i) {
                double g = self.grad_ptr()[i];
                a.grad_ptr()[i] += g;
                
                if (broadcast_b) {
                    b.grad_ptr()[i % N] += g; 
                } else {
                    b.grad_ptr()[i] += g;
                }
            }
        });
        return res;
    }

    Tensor operator-(const Tensor& a, const Tensor& b) {
        Tensor res(a.shape(), false);
        for(size_t i=0; i<a.size(); ++i) res.item(i) = a.item(i) - b.item(i);

        res.set_backward_fn({a, b}, [a, b](Tensor& self) mutable {
            for(size_t i=0; i<self.size(); ++i) {
                double g = self.grad_ptr()[i];
                a.grad_ptr()[i] += g;
                b.grad_ptr()[i] -= g;
            }
        });
        return res;
    }

    Tensor operator*(const Tensor& a, const Tensor& b) {
        Tensor res(a.shape(), false);
        for(size_t i=0; i<a.size(); ++i) res.item(i) = a.item(i) * b.item(i);

        res.set_backward_fn({a, b}, [a, b](Tensor& self) mutable {
            for(size_t i=0; i<self.size(); ++i) {
                double g = self.grad_ptr()[i];
                a.grad_ptr()[i] += g * b.item(i);
                b.grad_ptr()[i] += g * a.item(i);
            }
        });
        return res;
    }

    Tensor matmul(const Tensor& a, const Tensor& b) {
        size_t M = a.shape()[0];
        size_t K = a.shape()[1];
        size_t N = b.shape()[1];
        
        Tensor res({M, N}, false);

        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                double sum = 0;
                for (size_t k = 0; k < K; k++) {
                    sum += a.item(i*K + k) * b.item(k*N + j);
                }
                res.item(i*N + j) = sum;
            }
        }

        res.set_backward_fn({a, b}, [a, b](Tensor& self) mutable {
            size_t M = a.shape()[0];
            size_t K = a.shape()[1];
            size_t N = b.shape()[1];

            for(size_t i=0; i<M; ++i) {
                for(size_t k=0; k<K; ++k) {
                    double sum = 0;
                    for(size_t j=0; j<N; ++j) {
                        sum += self.grad_ptr()[i*N + j] * b.item(k*N + j);
                    }
                    a.grad_ptr()[i*K + k] += sum;
                }
            }

            for(size_t k=0; k<K; ++k) {
                for(size_t j=0; j<N; ++j) {
                    double sum = 0;
                    for(size_t i=0; i<M; ++i) {
                        sum += a.item(i*K + k) * self.grad_ptr()[i*N + j];
                    }
                    b.grad_ptr()[k*N + j] += sum;
                }
            }
        });
        return res;
    }

    Tensor Tensor::relu(const Tensor& input) {
        Tensor res(input.shape(), false);
        for(size_t i=0; i<input.size(); ++i) 
            res.item(i) = std::max(0.0, input.item(i));

        res.set_backward_fn({input}, [input](Tensor& self) mutable {
            for(size_t i=0; i<self.size(); ++i) {
                if (input.item(i) > 0) {
                    input.grad_ptr()[i] += self.grad_ptr()[i];
                }
            }
        });
        return res;
    }

    Tensor Tensor::pow(const Tensor& base, double exp) {
        Tensor res(base.shape(), false);
        for(size_t i=0; i<base.size(); ++i) 
            res.item(i) = std::pow(base.item(i), exp);

        res.set_backward_fn({base}, [base, exp](Tensor& self) mutable {
            for(size_t i=0; i<self.size(); ++i) {
                double d = exp * std::pow(base.item(i), exp - 1.0);
                base.grad_ptr()[i] += self.grad_ptr()[i] * d;
            }
        });
        return res;
    }

    Tensor Tensor::sum() const {
        double total = 0;
        if (impl) for(auto v : impl->data) total += v;
        Tensor res = Tensor(total); 

        res.set_backward_fn({*this}, [input = *this](Tensor& self) mutable {
            double g = self.grad_ptr()[0]; 
            for(size_t i=0; i<input.size(); ++i) {
                input.grad_ptr()[i] += g; 
            }
        });
        return res;
    }

    void print(const Tensor& t) {
        std::cout << "Tensor(" << t.shape()[0] << "x" << (t.shape().size() > 1 ? t.shape()[1] : 1) << ")\n";
    }
}