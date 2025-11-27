// include/axon/Optimizer.hpp
// github.com/51ddhesh/axon
// MIT License

#ifndef AXON_OPTIMIZER_HPP
#define AXON_OPTIMIZER_HPP

#include "axon/Tensor.hpp"  

namespace axon::optim {
    class Optimizer {
    protected:
        std::vector<Tensor> params_;
        double lr_;

    public:
        Optimizer(std::vector<Tensor>& params, double lr);
        virtual ~Optimizer() = default;

        void zero_grad();
        virtual void step() = 0;
    };

    class SGD : public Optimizer {
    public:
        SGD(std::vector<Tensor> params, double lr);
        void step() override;
    };
} // namespace axon::optim

#endif // AXON_OPTIMIZER_HPP
