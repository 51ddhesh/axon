#include "axon3/tensor.hpp"
#include "axon3/ops.hpp"
#include "axon3/optimizer.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace axon;

int main() {
    std::cout << "[TEST] Self-Attention Training Step (Integration)...\n";

    // Configuration
    int B = 2; // Batch
    int S = 4; // Sequence
    int D = 8; // Embedding Dim
    int d_k = 8; // Head Dim

    // Input: (B, S, D)
    Tensor x = Tensor::zeros({B, S, D}); // Dummy data
    // Fill with random
    for(size_t i=0; i<x.numel(); i++) x.data_ptr()[i] = (float)rand()/RAND_MAX;

    // Weights for Q, K, V (Linear layers essentially)
    // For simplicity, we assume 1 Head, so projection is (D, D)
    Tensor Wq = Tensor::ones({D, d_k});
    Tensor Wk = Tensor::ones({D, d_k});
    Tensor Wv = Tensor::ones({D, d_k});

    Wq.set_requires_grad(true);
    Wk.set_requires_grad(true);
    Wv.set_requires_grad(true);

    // Optimizer
    AdamW optim({Wq, Wk, Wv}, 0.01f);

    std::cout << "  1. Forward Pass...\n";

    // 1. Projections
    // Q = X @ Wq, K = X @ Wk, V = X @ Wv
    // Shapes: (B, S, D) @ (D, d_k) -> (B, S, d_k)
    // Note: Our matmul automatically broadcasts (D, d_k) to (B, D, d_k)
    Tensor Q = matmul(x, Wq);
    Tensor K = matmul(x, Wk);
    Tensor V = matmul(x, Wv);

    // 2. Scaled Dot-Product Attention
    // Scores = Q @ K.T / sqrt(d_k)
    // K shape: (B, S, d_k). Need K.T -> (B, d_k, S)
    // Since K is rank 3, transpose(K, 1, 2) swaps last two dims.
    Tensor K_T = transpose(K, 1, 2);
    
    Tensor scores = matmul(Q, K_T); // (B, S, S)
    
    // Scale
    float scale = 1.0f / std::sqrt((float)d_k);
    Tensor scale_t = Tensor::zeros({1}); scale_t.data_ptr()[0] = scale;
    scores = mul(scores, scale_t);

    // Softmax (applied on last dim S)
    Tensor attn_weights = softmax(scores);

    // Output = Attn @ V
    // (B, S, S) @ (B, S, d_k) -> (B, S, d_k)
    Tensor context = matmul(attn_weights, V);

    // 3. Loss (Dummy: Sum of output)
    Tensor loss = sum(context);

    std::cout << "     Loss: " << loss.at({0}) << "\n";

    std::cout << "  2. Backward Pass...\n";
    optim.zero_grad();
    loss.backward();

    assert(Wq.get_grad() != nullptr);
    assert(Wk.get_grad() != nullptr);
    assert(Wv.get_grad() != nullptr);

    std::cout << "  3. Optimizer Step (AdamW)...\n";
    float w_before = Wq.at({0,0});
    optim.step();
    float w_after = Wq.at({0,0});

    if (w_before == w_after) {
        std::cerr << "Optimizer Error: Weights did not change.\n";
        return 1;
    }
    
    std::cout << "     Weight updated: " << w_before << " -> " << w_after << "\n";
    std::cout << "  -> Integration Passed.\n";

    return 0;
}