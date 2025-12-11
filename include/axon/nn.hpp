#pragma once

#include "tensor.hpp"
#include "ops.hpp"
#include <cmath>
#include <random>
#include <vector>

namespace axon::nn {
    // the base module
    // similar to `torch.nn.Module`
    struct Module { 
        virtual ~Module() = default;
        virtual Tensor forward(Tensor x) = 0;
        virtual std::vector<Tensor> parameters() = 0;
        
        
        // training mode flag for Dropout/BatchNorm later on
        bool is_training = true;
        
        void train() {
            is_training = true;
        }

        void eval() {
            is_training = false;
        }
    };

    class Linear : public Module {
    public:
        Tensor weight;
        Tensor bias;

        Linear(int in_features, int out_features)
            : weight(Tensor::zeros({in_features, out_features})),
              bias(Tensor::zeros({1, out_features})) {
        
            float limit = std::sqrt(6.0f / (float)(in_features + out_features));

            float* w_data = weight.data_ptr();
            size_t w_size = weight.numel();
            float r;
            for (size_t i = 0; i < w_size; i++) {
                r = (float)rand() / RAND_MAX;
                w_data[i] = (r * 2 * limit) - limit;
            }

            weight.set_requires_grad(true);
            bias.set_requires_grad(true);
        }

        Tensor forward(Tensor x) override {
            // y = x @ w + b
            return axon::add(axon::matmul(x, weight), bias);
        }

        std::vector<Tensor> parameters() override {
            return {weight, bias};
        }
    };

    class Embedding : public Module {
    public:
        Tensor weight;
        Embedding(int num_embeddings, int embedding_dims) :
            weight(Tensor::zeros({num_embeddings, embedding_dims})) {
            
            float* d = weight.data_ptr();
            size_t w_size = weight.numel();

            for(size_t i = 0; i < w_size; i++) {
                d[i] = (float)(rand() / RAND_MAX) - 0.5f;
            }
            weight.set_requires_grad(true);
        }

        Tensor forward(Tensor x) override {
            return axon::embedding(x, weight);
        }

        std::vector<Tensor> parameters() override {
            return { weight };
        }
    };

    class LayerNorm : public Module {
    public:
        Tensor gamma;
        Tensor beta;
        float eps;

        LayerNorm(int normalized_shape, float eps = 1e-5) 
            : gamma(Tensor::ones({normalized_shape})), 
              beta(Tensor::zeros({normalized_shape})), 
              eps(eps) {
            
            gamma.set_requires_grad(true);
            beta.set_requires_grad(true);
        }

        Tensor forward(Tensor x) override {
            return axon::layer_norm(x, gamma, beta, eps);
        }

        std::vector<Tensor> parameters() override {
            return {gamma, beta};
        }
    };

    class MultiHeadAttention : public Module {
    public:
        Linear w_q, w_k, w_v;
        Linear c_proj;
        int n_head;
        int head_dim;

        MultiHeadAttention(int n_embd, int n_head) :
            w_q(n_embd, n_embd), w_k(n_embd, n_embd), w_v(n_embd, n_embd),
            c_proj(n_embd, n_embd), n_head(n_head), head_dim(n_embd / n_head) {}

        Tensor forward(Tensor x) override {
            int B = x.get_shape()[0];
            int T = x.get_shape()[1];

            Tensor q = w_q.forward(x);
            Tensor k = w_k.forward(x);
            Tensor v = w_v.forward(x);

            q = axon::view(q, {B, T, n_head, head_dim});
            k = axon::view(k, {B, T, n_head, head_dim});
            v = axon::view(v, {B, T, n_head, head_dim});
        
            q = axon::transpose(q, 1, 2);
            k = axon::transpose(k, 1, 2);
            v = axon::transpose(v, 1, 2);

            Tensor k_t = axon::transpose(k, 2, 3);
            Tensor scores = axon::matmul(q, k_t);

            float scale = 1.0f / std::sqrt((float)head_dim);
            Tensor scale_t = Tensor::zeros({1});
            scale_t.data_ptr()[0] = scale;
            scores = axon::mul(scores, scale_t);
            
            Tensor mask = Tensor::zeros({T, T});
            float* m_ptr = mask.data_ptr();

            float neg_inf = -1e9f;

            for (int i = 0; i < T; i++) {
                for (int j = 0; j < T; j++) {
                    if (j > i) {
                        m_ptr[i * T + j] = neg_inf;
                    }
                }
            }

            scores = axon::add(scores, mask);

            Tensor attn = axon::softmax(scores);

            Tensor context = axon::matmul(attn, v);

            context = axon::transpose(context, 1, 2);

            context = axon::view(context, {B, T, n_head * head_dim});

            return c_proj.forward(context);
        }

        std::vector<Tensor> parameters() override {
            std::vector<Tensor> params;
            auto p_q = w_q.parameters(); params.insert(params.end(), p_q.begin(), p_q.end());
            auto p_k = w_k.parameters(); params.insert(params.end(), p_k.begin(), p_k.end());
            auto p_v = w_v.parameters(); params.insert(params.end(), p_v.begin(), p_v.end());
            auto p_c = c_proj.parameters(); params.insert(params.end(), p_c.begin(), p_c.end());
            return params;
        }
    };

    class FeedForward : public Module {
    public:
        Linear c_fc;
        Linear c_proj;

        FeedForward(int n_embd) :
            c_fc(n_embd, 4 * n_embd),
            c_proj(4 * n_embd, n_embd) {}

        Tensor forward(Tensor x) override {
            Tensor h = c_fc.forward(x);
            h = axon::gelu(h);
            return c_proj.forward(h);
        }

        std::vector<Tensor> parameters() override {
            auto p1 = c_fc.parameters();
            auto p2 = c_proj.parameters();
            p1.insert(p1.end(), p2.begin(), p2.end());
            return p1;
        }
    }; 

    class Block : public Module {
    public:
        LayerNorm ln_1;
        MultiHeadAttention attn;
        LayerNorm ln_2;
        FeedForward mlp;

        Block(int n_embd, int n_head) :
            ln_1(n_embd), attn(n_embd, n_head),
            ln_2(n_embd), mlp(n_embd) {}
    
        Tensor forward(Tensor x) override {
            // GPT-2 Architecture: Pre-Norm
            // 1. Attention Block: x = x + attn(ln1(x))
            Tensor h1 = ln_1.forward(x);
            Tensor attn_out = attn.forward(h1);
            x = axon::add(x, attn_out);

            // 2. MLP Block: x = x + mlp(ln2(x))
            Tensor h2 = ln_2.forward(x);
            Tensor mlp_out = mlp.forward(h2);
            x = axon::add(x, mlp_out);

            return x;
        }

        std::vector<Tensor> parameters() override {
            std::vector<Tensor> params;
            auto p_l1 = ln_1.parameters(); params.insert(params.end(), p_l1.begin(), p_l1.end());
            auto p_att = attn.parameters(); params.insert(params.end(), p_att.begin(), p_att.end());
            auto p_l2 = ln_2.parameters(); params.insert(params.end(), p_l2.begin(), p_l2.end());
            auto p_mlp = mlp.parameters(); params.insert(params.end(), p_mlp.begin(), p_mlp.end());
            return params;
        }
    };

    class GPT2 : public Module {
    public:
        Embedding wte; // Token Embeddings
        Embedding wpe; // Position Embeddings
        std::vector<Block> h; // Transformer Blocks
        LayerNorm ln_f; // Final LayerNorm
        Linear lm_head; // Language Model Head (Projects back to Vocab)

        // GPT-2 Small Config:
        // Vocab: 50257
        // Context: 1024
        // Dim: 768
        // Layers: 12
        // Heads: 12
        GPT2() 
            : wte(50257, 768), 
              wpe(1024, 768), 
              ln_f(768), 
              lm_head(768, 50257) {
            
            // Create 12 Blocks
            // We use reserve to avoid reallocation issues during construction
            h.reserve(12);
            for(int i = 0; i < 12; ++i) {
                h.emplace_back(768, 12);
            }
        }

        Tensor forward(Tensor idx) override {
            // idx: (Batch, Seq) of Integer Tokens
            int B = idx.get_shape()[0];
            int T = idx.get_shape()[1];

            // 1. Token Embeddings
            Tensor tok_emb = wte.forward(idx); // (B, T, 768)

            // 2. Position Embeddings
            // Create position indices [0, 1, 2, ... T-1]
            Tensor pos_idx = Tensor::zeros({T});
            for(int i=0; i<T; ++i) pos_idx.data_ptr()[i] = (float)i;
            
            // Expand to batch (B, T) if necessary, or let broadcasting handle it.
            // Axon broadcasting: (B, T, C) + (T, C) works fine.
            Tensor pos_emb = wpe.forward(pos_idx); 

            Tensor x = axon::add(tok_emb, pos_emb);

            // 3. Blocks
            for(auto& block : h) {
                x = block.forward(x);
            }

            // 4. Final Norm
            x = ln_f.forward(x);

            // 5. LM Head (Logits)
            // (B, T, 768) @ (768, 50257) -> (B, T, 50257)
            return lm_head.forward(x);
        }

        std::vector<Tensor> parameters() override {
            std::vector<Tensor> params;
            // Order is CRITICAL for loading!
            // 1. WTE
            params.push_back(wte.weight);
            // 2. WPE
            params.push_back(wpe.weight);
            // 3. Blocks
            for(auto& block : h) {
                auto p = block.parameters();
                params.insert(params.end(), p.begin(), p.end());
            }
            // 4. Final LN
            params.push_back(ln_f.gamma);
            params.push_back(ln_f.beta);
            // 5. LM Head
            params.push_back(lm_head.weight);
            params.push_back(lm_head.bias);
            
            return params;
        }
    };
}
