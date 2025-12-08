#include "axon3/tensor.hpp"
#include "axon3/ops.hpp"
#include "axon3/nn.hpp"
#include "axon3/optimizer.hpp"
#include "mnist.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

int main() {
    // 1. Load Data
    // Note: Ensure you have downloaded MNIST files into a 'data' folder!
    std::string img_path = "data/train-images.idx3-ubyte";
    std::string lbl_path = "data/train-labels.idx1-ubyte";
    
    // Load just 1000 images for a quick test run. Remove argument to load all 60k.
    axon::MNISTData data = axon::load_mnist(img_path, lbl_path, 2000);

    // 2. Define Model
    // 784 (Pixels) -> 128 (Hidden) -> 10 (Classes)
    axon::nn::Linear fc1(784, 128);
    axon::nn::Linear fc2(128, 10);

    // 3. Optimizer
    std::vector<axon::Tensor> params;
    auto p1 = fc1.parameters();
    auto p2 = fc2.parameters();
    params.insert(params.end(), p1.begin(), p1.end());
    params.insert(params.end(), p2.begin(), p2.end());
    
    axon::SGD optimizer(params, 0.01f); // Learning Rate

    // 4. Training Loop
    int batch_size = 32;
    int num_samples = data.images.get_shape()[0];
    int epochs = 3;

    std::cout << "--- Starting Training ---\n";
    std::cout << std::fixed << std::setprecision(4);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int batches = 0;

        for (int i = 0; i < num_samples; i += batch_size) {
            int current_batch_size = std::min(batch_size, num_samples - i);
            
            axon::Tensor x_batch = axon::Tensor::zeros({current_batch_size, 784});
            axon::Tensor y_batch = axon::Tensor::zeros({current_batch_size, 10});
            
            // Copy data
            float* src_x = data.images.data_ptr() + i * 784;
            float* dst_x = x_batch.data_ptr();
            std::memcpy(dst_x, src_x, current_batch_size * 784 * sizeof(float));
            
            float* src_y = data.labels.data_ptr() + i * 10;
            float* dst_y = y_batch.data_ptr();
            std::memcpy(dst_y, src_y, current_batch_size * 10 * sizeof(float));
            
            // --- Forward ---
            auto h1 = axon::relu(fc1.forward(x_batch));
            auto logits = fc2.forward(h1);
            auto log_probs = axon::log_softmax(logits);
            auto loss = axon::nll_loss(log_probs, y_batch);

            // --- Backward ---
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            total_loss += loss.at({0});
            batches++;
        }
        
        std::cout << "Epoch " << epoch << " | Avg Loss: " << (total_loss / batches) << "\n";
    }

    // 5. Inference Check (First 5 images)
    std::cout << "\n--- Inference Check ---\n";
    for(int i=0; i<5; i++) {
        // Grab one image
        axon::Tensor img = axon::Tensor::zeros({1, 784});
        std::memcpy(img.data_ptr(), data.images.data_ptr() + i*784, 784*sizeof(float));
        
        // Forward
        auto h1 = axon::relu(fc1.forward(img));
        auto probs = axon::log_softmax(fc2.forward(h1));
        
        // Find Argmax manually
        float max_val = -1e9;
        int pred = -1;
        float* p = probs.data_ptr();
        for(int c=0; c<10; c++) {
            if(p[c] > max_val) { max_val = p[c]; pred = c; }
        }
        
        // Find Target
        int target = -1;
        float* t = data.labels.data_ptr() + i*10;
        for(int c=0; c<10; c++) { if(t[c] > 0.9f) target = c; }

        std::cout << "Image " << i << ": Pred=" << pred << " | Target=" << target << "\n";
    }

    return 0;
}