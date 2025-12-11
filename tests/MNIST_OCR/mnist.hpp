#pragma once
#include "axon/tensor.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cassert>

namespace axon {

    // Helper to read Big Endian integers from MNIST files
    uint32_t read_header(std::ifstream& file) {
        uint32_t val = 0;
        file.read(reinterpret_cast<char*>(&val), 4);
        // Swap endianness (Intel/AMD are Little Endian, MNIST is Big Endian)
        return ((val << 24) & 0xFF000000) |
               ((val << 8)  & 0x00FF0000) |
               ((val >> 8)  & 0x0000FF00) |
               ((val >> 24) & 0x000000FF);
    }

    struct MNISTData {
        Tensor images; // (Batch, 784)
        Tensor labels; // (Batch, 10) - One Hot
    };

    MNISTData load_mnist(const std::string& image_path, const std::string& label_path, int limit = -1) {
        std::ifstream f_img(image_path, std::ios::binary);
        std::ifstream f_lbl(label_path, std::ios::binary);

        if (!f_img.is_open() || !f_lbl.is_open()) {
            throw std::runtime_error("Could not open MNIST files. Check paths!");
        }

        // --- Read Image Header ---
        uint32_t magic_img = read_header(f_img); // 2051
        uint32_t num_imgs  = read_header(f_img);
        uint32_t rows      = read_header(f_img);
        uint32_t cols      = read_header(f_img);
        
        // --- Read Label Header ---
        uint32_t magic_lbl = read_header(f_lbl); // 2049
        uint32_t num_lbls  = read_header(f_lbl);

        assert(num_imgs == num_lbls);
        
        // Limit dataset size for quicker debugging/testing
        int N = (limit > 0 && limit < (int)num_imgs) ? limit : (int)num_imgs;
        int dim = rows * cols; // 28*28 = 784

        std::cout << "Loading " << N << " samples from MNIST...\n";

        // --- Allocate Tensors ---
        Tensor X = Tensor::zeros({N, dim});
        Tensor Y = Tensor::zeros({N, 10}); // 10 Classes

        float* x_ptr = X.data_ptr();
        float* y_ptr = Y.data_ptr();

        // --- Read Data ---
        // Buffers
        std::vector<uint8_t> pixel_buf(dim);
        uint8_t label_buf;

        for(int i = 0; i < N; ++i) {
            // Read 1 Image
            f_img.read(reinterpret_cast<char*>(pixel_buf.data()), dim);
            // Read 1 Label
            f_lbl.read(reinterpret_cast<char*>(&label_buf), 1);

            // Process Image (Normalize 0-255 -> 0.0-1.0)
            for(int p = 0; p < dim; ++p) {
                x_ptr[i * dim + p] = static_cast<float>(pixel_buf[p]) / 255.0f;
            }

            // Process Label (One-Hot Encode)
            // e.g., label 2 -> [0, 0, 1, 0, ...]
            if (label_buf < 10) {
                y_ptr[i * 10 + label_buf] = 1.0f;
            }
        }

        return {X, Y};
    }
}