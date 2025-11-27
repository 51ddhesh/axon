#include "axon/Optimizer.hpp"

namespace axon::optim {

    Optimizer::Optimizer(std::vector<Tensor>& params, double lr) 
        : params_(params), lr_(lr) {}

    void Optimizer::zero_grad() {
        for (auto& p : params_) {
            p.zero_grad();
        }
    }

    SGD::SGD(std::vector<Tensor> params, double lr) 
        : Optimizer(params, lr) {}

    void SGD::step() {
        for (auto& p : params_) {
            // W = W - lr * grad
            // We modify data directly (In-Place Update)
            
            double* data = p.data_ptr();
            double* grad = p.grad_ptr();
            size_t size = p.size();

            // Simple SGD
            for (size_t i = 0; i < size; ++i) {
                data[i] -= lr_ * grad[i];
            }
        }
    }

} // namespace axon::optim