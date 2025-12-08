#include "axon3/tensor.hpp"
#include "axon3/nn.hpp"
#include "axon3/grad_mode.hpp" 
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

int argmax(const float* ptr, size_t size) {
    int max_idx = 0;
    float max_val = ptr[0];
    for(size_t i = 1; i < size; ++i) {
        if(ptr[i] > max_val) {
            max_val = ptr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

int main() {
    axon::nn::GPT2 model;
    std::cout << "[Axon] Loading GPT-2 124M weights...\n";
    
    try {
        auto params = model.parameters();
        axon::load_model(params, "gpt2_axon.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[Axon] Weights loaded.\n\n";

    // "The capital of France is"
    std::vector<int> input_ids = {464, 3225, 286, 4881, 318};
    
    int max_new_tokens = 5; 
    
    std::cout << "Prompt IDs: ";
    for(int id : input_ids) std::cout << id << " ";
    std::cout << "\n[Axon] Generating...\n";

    axon::NoGradGuard guard; 

    for (int i = 0; i < max_new_tokens; ++i) {
        int seq_len = input_ids.size();
        axon::Tensor input = axon::Tensor::zeros({1, seq_len});
        
        float* ptr = input.data_ptr();
        for(int j=0; j<seq_len; ++j) {
            ptr[j] = static_cast<float>(input_ids[j]);
        }

        axon::Tensor logits = model.forward(input); 

        size_t vocab_size = 50257;
        // Last token logits
        float* last_token_logits = logits.data_ptr() + (seq_len - 1) * vocab_size;

        int next_token = argmax(last_token_logits, vocab_size);
        
        input_ids.push_back(next_token);
        
        std::cout << next_token << " " << std::flush;
    }
    
    std::cout << "\n\nResult IDs: [ ";
    for(int id : input_ids) std::cout << id << ", ";
    std::cout << "]\n";

    return 0;
}