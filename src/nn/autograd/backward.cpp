// src/nn/autograd/backward.cpp

#include "../../../include/Tensor.hpp"  
#include <stdexcept>
#include <numeric>
#include <vector>
#include <algorithm>

// Helper function to perform topological sort using DFS
void Tensor::build_topo(std::vector<Tensor*>& sorted, std::unordered_set<Tensor*>& visited) {
    if (visited.find(this) == visited.end()) {
        visited.insert(this);
        for (Tensor* child : this -> _prev) {
            child -> build_topo(sorted, visited);
        }
        sorted.push_back(this);
    }
}

void Tensor::backward() {
    std::vector<Tensor*> topo_sorted;
    std::unordered_set<Tensor*> visited;

    this -> build_topo(topo_sorted, visited);
 
    this -> _grad = std::make_shared<Tensor>(Tensor::ones(this -> rows(), this -> cols()));

    std::reverse(topo_sorted.begin(), topo_sorted.end());

    for (Tensor* node : topo_sorted) {
        for (Tensor* parent : node->_prev) {
            if (!parent -> _grad) {
                parent -> _grad = std::make_shared<Tensor>(Tensor::zeros(parent->rows(), parent->cols()));
            }
        }
        if (node -> _backward_fn) {
            node -> _backward_fn(node);
        }
    }
}

