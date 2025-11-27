// include/axon/NN.hpp 
// github.com/51ddhesh/axon
// MIT License

#ifndef AXON_NN_HPP
#define AXON_NN_HPP

#include "axon/Tensor.hpp"  

namespace axon::nn {
    class Module {
    public:
        virtual ~Module() = default;
        virtual std::vector<Tensor> parameters() = 0;
    };

    class Linear : public Module {
    private:
        Tensor W;
        Tensor b;
        bool use_bias;

        void reset_parameters(size_t in_features, size_t out_features);

    public:
        Linear(size_t in_features, size_t out_features, bool bias = true);
        
        Tensor operator() (const Tensor& input);

        std::vector<Tensor> parameters() override;
    };
} // namespace axon::nn

#endif // AXON_NN_HPP
