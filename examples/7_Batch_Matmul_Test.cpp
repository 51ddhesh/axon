#include "axon/tensor.hpp"
#include "axon/ops.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>

using namespace axon;

void test_simple_batch() {
    std::cout << "[TEST] Simple Batch MatMul (B, M, K) @ (B, K, N)...\n";
    
    // Shape: (2, 2, 3) @ (2, 3, 2) -> Output (2, 2, 2)
    // We use all Ones. Result should be K (3.0) everywhere.
    Tensor A = Tensor::ones({2, 2, 3});
    Tensor B = Tensor::ones({2, 3, 2});
    
    Tensor C = axon::matmul(A, B);

    // 1. Check Shape
    assert(C.get_shape().size() == 3);
    assert(C.get_shape()[0] == 2); // Batch
    assert(C.get_shape()[1] == 2); // M
    assert(C.get_shape()[2] == 2); // N

    // 2. Check Values
    float* ptr = C.data_ptr();
    for(size_t i=0; i<C.numel(); ++i) {
        if(ptr[i] != 3.0f) {
            std::cerr << "Value mismatch at " << i << ". Expected 3.0, got " << ptr[i] << "\n";
            exit(1);
        }
    }
    std::cout << "  -> Shapes and Basic Values Passed.\n";
}

void test_pointer_offset_logic() {
    std::cout << "[TEST] Batch Pointer Offsets (CRITICAL)...\n";
    
    // This tests if the loop actually moves memory pointers for Batch 1.
    // Batch 0: Matrices of 1s. Expected Result: 1*1*K = 3
    // Batch 1: Matrices of 2s. Expected Result: 2*2*K = 12
    
    Tensor A = Tensor::zeros({2, 2, 3});
    Tensor B = Tensor::zeros({2, 3, 2});

    // Fill Batch 0 with 1s
    // A has 2*2*3 = 12 elements total. Batch 0 is first 6.
    for(int i=0; i<6; i++) A.data_ptr()[i] = 1.0f;
    for(int i=0; i<6; i++) B.data_ptr()[i] = 1.0f;

    // Fill Batch 1 with 2s
    for(int i=6; i<12; i++) A.data_ptr()[i] = 2.0f;
    for(int i=6; i<12; i++) B.data_ptr()[i] = 2.0f;

    Tensor C = axon::matmul(A, B);

    // Check Batch 0 Result (indices 0-3 in output)
    for(int i=0; i<4; i++) {
        assert(C.data_ptr()[i] == 3.0f);
    }

    // Check Batch 1 Result (indices 4-7 in output)
    for(int i=4; i<8; i++) {
        if (C.data_ptr()[i] != 12.0f) {
            std::cerr << "OFFSET ERROR! Batch 1 calculation used Batch 0 data?\n";
            std::cerr << "Expected 12.0, got " << C.data_ptr()[i] << "\n";
            exit(1);
        }
    }
    std::cout << "  -> Pointer Arithmetic Valid (Batches are independent).\n";
}

void test_transformer_broadcast() {
    std::cout << "[TEST] 4D Transformer Shapes (Broadcasting)...\n";

    // Scenario: Attention Scores
    // Q: (Batch=2, Heads=2, Seq=4, Dim=8)
    // K: (Batch=2, Heads=2, Dim=8, Seq=4)
    // Out: (2, 2, 4, 4)
    
    Tensor Q = Tensor::ones({2, 2, 4, 8});
    Tensor K = Tensor::ones({2, 2, 8, 4});

    // Let's pretend one head has different values to test complex strides
    // Not strictly needed if previous test passed, but good sanity check.
    
    Tensor Scores = axon::matmul(Q, K);
    
    assert(Scores.get_shape().size() == 4);
    assert(Scores.get_shape()[2] == 4);
    assert(Scores.get_shape()[3] == 4);
    
    // Value check: 1 * 1 * Dim(8) = 8.0f
    assert(Scores.at({0,0,0,0}) == 8.0f);

    std::cout << "  -> 4D Shapes Passed.\n";
}

void test_batch_autograd() {
    std::cout << "[TEST] Batched Autograd...\n";
    
    // A: (2, 2, 3)
    // B: (2, 3, 2)
    // Loss = Sum(A @ B)
    // Backward check.
    
    Tensor A = Tensor::ones({2, 2, 3});
    Tensor B = Tensor::ones({2, 3, 2});
    A.set_requires_grad(true);

    Tensor C = axon::matmul(A, B);
    Tensor loss = axon::sum(C);
    
    loss.backward();

    // Check Grad A
    // dL/dA = (dL/dC) @ B.T
    // dL/dC is ones(2, 2, 2). B.T is ones(2, 2, 3).
    // Matrix mult: (2,2) @ (2,3) -> (2,3)
    // Since everything is 1s:
    // grad_A entry = (row of ones) dot (col of ones from B.T)
    // B has shape (3, 2). B.T has shape (2, 3).
    // Summing over the dim of size 2.
    // Expected value = 2.0f
    
    assert(A.get_grad() != nullptr);
    Tensor g = *A.get_grad();
    
    assert(g.get_shape()[0] == 2);
    assert(g.get_shape()[2] == 3);
    
    if (std::abs(g.at({0,0,0}) - 2.0f) > 0.001f) {
        std::cerr << "Grad Mismatch. Expected 2.0f, got " << g.at({0,0,0}) << "\n";
        exit(1);
    }

    std::cout << "  -> Autograd Passed.\n";
}

int main() {
    test_simple_batch();
    test_pointer_offset_logic();
    test_transformer_broadcast();
    test_batch_autograd();
    std::cout << "---------------------------------\n";
    std::cout << "ALL BATCH MATMUL TESTS PASSED.\n";
}