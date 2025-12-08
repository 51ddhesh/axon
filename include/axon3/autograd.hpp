#pragma once

#include "axon3/tensor.hpp"
#include <memory>
#include <vector>

namespace axon {
    class Tensor;

    struct GradFn {
        struct Edge {
            std::shared_ptr<GradFn> fn;
            std::shared_ptr<Tensor> input_tensor;
        };

        std::vector<Edge> next_edges;
        virtual ~GradFn() = default;

        virtual std::vector<Tensor> apply(const Tensor& grad_output) = 0;
    };

    
} // namespace axon
