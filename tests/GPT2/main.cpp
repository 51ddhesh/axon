#include "axon/tensor.hpp"
#include "axon/nn.hpp"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

// Helper to find the index of the maximum value (Greedy Decoding)
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
    // 1. Initialize Model
    axon::nn::GPT2 model;
    std::cout << "Created GPT-2 Model.\n";
    
    std::cout << "Loading weights from gpt2_axon.bin ...\n";
    try {
        auto params = model.parameters();
        axon::load_model(params, "gpt2_axon.bin");
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }
    std::cout << "Weights loaded successfully!\n\n";

    // 2. Prepare Input
    // Prompt: "The capital of France is"
    std::vector<int> input_ids = {464, 3225, 286, 4881, 318};
    
    int max_new_tokens = 10;
    
    std::cout << "Prompt IDs: ";
    for(int id : input_ids) std::cout << id << " ";
    std::cout << "\nGenerating " << max_new_tokens << " tokens...\n";
    std::cout << "--------------------------------------------------\n";

    // 3. Generation Loop
    for (int i = 0; i < max_new_tokens; ++i) {
        // Create Input Tensor (Batch=1, Seq=Current Length)
        int seq_len = input_ids.size();
        axon::Tensor input = axon::Tensor::zeros({1, seq_len});
        
        // Fill data
        float* ptr = input.data_ptr();
        for(int j=0; j<seq_len; ++j) {
            ptr[j] = static_cast<float>(input_ids[j]);
        }

        // Forward Pass
        // Note: For a real optimization, we would cache Key/Values (KV-Cache).
        // Here we re-compute everything for simplicity (slower but correct).
        axon::Tensor logits = model.forward(input); // Output: (1, Seq, 50257)

        // Get logits for the LAST token only
        // Offset = (Batch=0) + (Seq-1) * VocabSize
        size_t vocab_size = 50257;
        float* last_token_logits = logits.data_ptr() + (seq_len - 1) * vocab_size;

        // Greedy Decode (Argmax)
        int next_token = argmax(last_token_logits, vocab_size);
        
        // Append to input for next iteration
        input_ids.push_back(next_token);
        
        std::cout << next_token << " " << std::flush;
    }
    
    std::cout << "\n--------------------------------------------------\n";
    std::cout << "Final Token Sequence: [ ";
    for(int id : input_ids) std::cout << id << ", ";
    std::cout << "]\n";

    return 0;
}