#include "axon3/tensor.hpp"
#include "axon3/nn.hpp"
#include <iostream>

int main() {
    axon::nn::GPT2 model;
    std::cout << "Created GPT-2 Model.\n";
    
    std::cout << "Loading weights...\n";
    auto params = model.parameters();
    axon::load_model(params, "gpt2_axon.bin");
    
    std::cout << "Weights loaded successfully!\n";
    
    // Quick Forward Check (Random Input)
    // Batch 1, Seq 5
    auto x = axon::Tensor::zeros({1, 5}); 
    x.data_ptr()[0] = 50256; // <|endoftext|>

    auto logits = model.forward(x);
    
    std::cout << "Forward pass complete. Logits Shape: ";
    for(int s : logits.get_shape()) std::cout << s << " ";
    std::cout << "\n";
    
    return 0;
}