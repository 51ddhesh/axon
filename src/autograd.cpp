#include "axon3/autograd.hpp"
#include "axon3/tensor.hpp"
#include "axon3/ops.hpp"
#include "axon3/grad_mode.hpp" 
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <stack>

namespace axon {

    bool GradMode::enabled = true;

    void Tensor::zero_grad() {
        set_grad(nullptr); 
    }

    void Tensor::add_grad(const Tensor& new_grad) {
        auto g = get_grad();
        if (!g) {
            // Create new gradient tensor if it doesn't exist
            set_grad(std::make_shared<Tensor>(new_grad.contiguous()));
        } else {
            // Accumulate: *g = *g + new_grad
            // Note: This creates a new storage for the result and updates *g to point to it.
            // Since state->grad is a shared_ptr to *g, the state sees the update.
            *g = axon::add(*g, new_grad);
        }
    }

    // Helper: Build Topological Order (Post-Order DFS)
    // We also map GradFn* back to the Tensor* that owns it, so we can retrieve gradients.
    void build_topo(Tensor* t, 
                    std::vector<GradFn*>& topo, 
                    std::unordered_set<GradFn*>& visited, 
                    std::unordered_map<GradFn*, Tensor*>& fn_to_tensor_map) {
        
        auto fn = t -> get_grad_fn().get();
        if (!fn || visited.count(fn)) return;

        visited.insert(fn);
        
        // Link the Fn back to the Tensor (so we can get t.grad later)
        fn_to_tensor_map[fn] = t;

        // Traverse dependencies first
        for (auto& edge : fn -> next_edges) {
            if (edge.input_tensor) {
                build_topo(edge.input_tensor.get(), topo, visited, fn_to_tensor_map);
            }
        }

        // Add to list (Post-Order)
        topo.push_back(fn);
    }

    void Tensor::backward() {
        // Seed the Gradient
        auto _grad = get_grad();
        if (!_grad) {
            if (numel() != 1) {
                // Warning logic...
            }
            _grad = std::make_shared<Tensor>(Tensor::ones(shape));
            set_grad(_grad);
        }

        // Build Graph Topology
        // We perform a DFS starting from 'this' (Loss).
        std::vector<GradFn*> topo_order;
        std::unordered_set<GradFn*> visited;
        std::unordered_map<GradFn*, Tensor*> fn_to_tensor_map;

        build_topo(this, topo_order, visited, fn_to_tensor_map);

        // Process in Reverse (Reverse Post-Order is Topological Order for Backprop)
        std::reverse(topo_order.begin(), topo_order.end());

        // Execute
        for (GradFn* fn : topo_order) {
            // Retrieve the gradient accumulated on the output tensor of this Fn
            Tensor* output_tensor = fn_to_tensor_map[fn];
            auto output_grad = output_tensor -> get_grad();

            // Safety: If grad is null (shouldn't happen in connected graph), treat as zeros
            if (!output_grad) continue;

            // Apply Chain Rule
            auto input_grads = fn -> apply(*output_grad);

            // Safety Check
            if (input_grads.size() != fn -> next_edges.size()) {
                 std::cerr << "!![FATAL] Autograd Graph Mismatch.\n";
                 std::terminate();
            }

            // Distribute to Inputs
            for (size_t i = 0; i < fn -> next_edges.size(); i++) {
                auto& edge = fn -> next_edges[i];
                if (edge.input_tensor && edge.input_tensor -> requires_grad()) {
                    edge.input_tensor -> add_grad(input_grads[i]);
                }
            }
        }
    }
}