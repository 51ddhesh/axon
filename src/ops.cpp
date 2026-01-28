#include "axon/ops.hpp"
#include "axon/kernels.hpp"    
#include "axon/autograd.hpp"
#include "axon/grad_mode.hpp"
#include <functional>
#include <stdexcept>
#include <algorithm>

namespace axon {

    std::vector<int> broadcast_shapes(const std::vector<int>& s1, const std::vector<int>& s2) {
        size_t len1 = s1.size();
        size_t len2 = s2.size();
        size_t max_len = std::max(len1, len2);
        std::vector<int> out_shape(max_len);

        for (size_t i = 0; i < max_len; i++) {
            int d1 = (i < len1) ? s1[len1 - 1 - i] : 1;
            int d2 = (i < len2) ? s2[len2 - 1 - i] : 1;

            if (d1 == d2) {
                out_shape[max_len - 1 - i] = d1;
            } else if (d1 == 1) {
                out_shape[max_len - 1 - i] = d2;
            } else if (d2 == 1) {
                out_shape[max_len - 1 - i] = d1;
            } else {
                throw std::invalid_argument("[BROADCASTING] Error: Shapes are incompatible");
            }
        }

        return out_shape;
    }

    void apply_binary_op_rec(
        int dim, const std::vector<int>& shape,
        int off_a, const std::vector<int>& stride_a,
        int off_b, const std::vector<int>& stride_b,
        int off_out, const std::vector<int>& stride_out,
        const float* ptr_a, const float* ptr_b, float* ptr_out,
        std::function<float(float, float)> op) {
        
        int dim_len = shape[dim];
        if (dim == shape.size() - 1) {
            for (int i = 0; i < dim_len; i++) {
                ptr_out[off_out + i * stride_out[dim]] =
                    op(ptr_a[off_a + i * stride_a[dim]], ptr_b[off_b + i * stride_b[dim]]);
            }
        } else {
            for (int i = 0; i < dim_len; i++) {
                apply_binary_op_rec(dim + 1, shape,
                    off_a + i * stride_a[dim], stride_a,
                    off_b + i * stride_b[dim], stride_b,
                    off_out + i * stride_out[dim], stride_out,
                    ptr_a, ptr_b, ptr_out, op);
            }
        }
    }

    void dispatch_binary_op(const Tensor& a, const Tensor& b, Tensor& out, std::function<float(float, float)> op) {
        apply_binary_op_rec(
            0, a.get_shape(),
            0, a.get_stride(),
            0, b.get_stride(),
            0, out.get_stride(),
            a.data_ptr(), b.data_ptr(), out.data_ptr(),
            op
        );
    }

    // Reduces `grad` to match `target_shape` by summing out broadcasted dimensions
    Tensor unbroadcast(Tensor grad, const std::vector<int>& target_shape) {
        
        // 1. Sum out extra leading dimensions (e.g. Batch broadcasting)
        // Grad: (32, 10), Target: (10) -> Sum dim 0
        while (grad.get_shape().size() > target_shape.size()) {
            grad = axon::sum(grad, 0, false); 
        }

        // 2. Sum out broadcasted singleton dimensions
        // Grad: (32, 10), Target: (32, 1) -> Sum dim 1, keep it (32, 1)
        for (size_t i = 0; i < target_shape.size(); ++i) {
            if (grad.get_shape()[i] != target_shape[i]) {
                if (target_shape[i] == 1) {
                    // This dim was broadcasted from 1 to N. Sum it back to 1.
                    grad = axon::sum(grad, i, true);
                } else {
                    throw std::runtime_error("[AUTOGRAD]: Gradient shape mismatch that cannot be unbroadcasted.");
                }
            }
        }
        return grad;
    }
    
    float sum_recursive(int dim, const Tensor& t, int offset) {
        float acc = 0.0f;
        int dim_len = t.get_shape()[dim];
        int stride = t.get_stride()[dim];

        if (dim == t.get_shape().size() - 1) {
            for (int i = 0; i < dim_len; i++) {
                acc += t.data_ptr()[offset + i * stride];
            }
        } else {
            for (int i = 0; i < dim_len; i++) {
                acc += sum_recursive(dim + 1, t, offset + i * stride);
            }
        }
        return acc;
    }

    struct ReluBackward : public GradFn {
        Tensor input;
        ReluBackward(Tensor input_tensor) : input(input_tensor) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            // We need the original input to compute the mask
            // But 'input' might be strided.
            Tensor grad_input = Tensor::zeros(input.get_shape());
            
            // Force contiguous for kernel execution
            Tensor inp_c = input.is_contiguous() ? input : input.contiguous();
            Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
            
            kernels::cpu::relu_backward_f32(inp_c.numel(), inp_c.data_ptr(), grad_out_c.data_ptr(), grad_input.data_ptr());
            
            return {
                grad_input
            };
        }
    };
    
    Tensor relu(Tensor t) {
        Tensor out = Tensor::zeros(t.get_shape());
        
        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        kernels::cpu::relu_f32(t_c.numel(), t_c.data_ptr(), out.data_ptr());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<ReluBackward>(t);
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }
        return out;
    }

    struct GeluBackward : public GradFn {
        Tensor input;
        GeluBackward(Tensor in) : input(in) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor grad_input = Tensor::zeros(input.get_shape());
            Tensor input_c = input.is_contiguous() ? input : input.contiguous();
            Tensor g_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
            kernels::cpu::gelu_backward_f32(input_c.numel(), input_c.data_ptr(), g_c.data_ptr(), grad_input.data_ptr());

            return {
                grad_input
            };
        }
    };

    Tensor gelu(Tensor t) {
        Tensor out = Tensor::zeros(t.get_shape());
        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        kernels::cpu::gelu_f32(t_c.numel(), t_c.data_ptr(), out.data_ptr());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<GeluBackward>(t);
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }

        return out;
    }

    struct LogSoftmaxBackward : public GradFn {
        Tensor output; 
        LogSoftmaxBackward(Tensor out) : output(out) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor grad_input = Tensor::zeros(output.get_shape());
            
            // Assume 2D (Batch, Class)
            int rows = output.get_shape()[0];
            int cols = output.get_shape()[1];
            
            Tensor out_c = output.is_contiguous() ? output : output.contiguous();
            Tensor g_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

            kernels::cpu::log_softmax_backward_f32(rows, cols, g_c.data_ptr(), out_c.data_ptr(), grad_input.data_ptr());
            return {grad_input};
        }
    };

    Tensor log_softmax(Tensor t) {
        if (t.get_shape().size() != 2) {
            throw std::invalid_argument("[DIM ERROR]: LogSoftmax expects 2D (Batch, Class)");
        }
        
        Tensor out = Tensor::zeros(t.get_shape());
        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        
        kernels::cpu::log_softmax_f32(t.get_shape()[0], t.get_shape()[1], t_c.data_ptr(), out.data_ptr());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<LogSoftmaxBackward>(out);
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }
        return out;
    }

    // NLL Loss (Negative Log Likelihood)
    // Expects LogSoftmax input.
    // Loss = - sum(target * input) / batch_size
    Tensor nll_loss(Tensor input, Tensor target) {
        // -1 * (target * input)
        Tensor prod = mul(target, input);
        Tensor s = sum(prod);
        Tensor neg_one = Tensor::ones({1}); neg_one.at({0}) = -1.0f;
        Tensor total_neg = mul(s, neg_one);
        
        // Mean
        Tensor batch_size = Tensor::ones({1}); 
        batch_size.at({0}) = 1.0f / (float)input.get_shape()[0];
        
        return mul(total_neg, batch_size);
    }

    struct ViewBackward : public GradFn {
        std::vector<int> original_shape;
        ViewBackward(std::vector<int> shape) : original_shape(shape) {}
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            return { 
                axon::view(grad_output, original_shape) 
            };
        }
    };

    Tensor view(Tensor t, const std::vector<int>& new_shape) {
        // Calculate size to verify compatibility
        size_t new_size = 1;
        for(int s : new_shape) new_size *= s;
        
        if (new_size != t.numel()) {
            throw std::invalid_argument("[VIEW] Size mismatch");
        }

        // Ensure Contiguity (View only works on contiguous memory conceptually)
        Tensor t_c = t.is_contiguous() ? t : t.contiguous();

        // Calculate Strides (Row-Major / C-Style)
        std::vector<int> new_stride(new_shape.size());
        int z = 1;
        for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
            new_stride[i] = z;  
            z *= new_shape[i];  
        }

        Tensor out = Tensor::from_storage(t_c.get_storage(), new_shape, new_stride, t_c.get_offset());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<ViewBackward>(t.get_shape());
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }
     
        return out;
    }

    struct PermuteBackward : public GradFn {
        std::vector<int> forward_dims;
        PermuteBackward(std::vector<int> dims) : forward_dims(dims) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            std::vector<int> argsort(forward_dims.size());
            for (size_t i = 0; i < forward_dims.size(); i++) {
                argsort[forward_dims[i]] = i;
            }
            return {
                axon::permute(grad_output, argsort)
            };
        }
    };

    Tensor permute(Tensor t, const std::vector<int>& dims) {
        if (dims.size() != t.get_shape().size()) {
            throw std::invalid_argument("[PERMUTE] Error: Dims mismatch");
        }

        std::vector<int> new_shape(dims.size());
        std::vector<int> new_stride(dims.size());

        auto old_shape = t.get_shape();
        auto old_stride = t.get_stride();

        for (size_t i = 0; i < dims.size(); i++) {
            new_shape[i] = old_shape[dims[i]];
            new_stride[i] = old_stride[dims[i]];    
        }

        Tensor out = Tensor::from_storage(t.get_storage(), new_shape, new_stride, t.get_offset());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<PermuteBackward>(dims);
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }

        return out;
    }

    struct AddBackward : public GradFn {
        // d(a + b)/da = 1, d(a + b)/db = 1
        std::vector<int> a_shape, b_shape;
        AddBackward(std::vector<int> sA, std::vector<int> sB) : a_shape(sA), b_shape(sB) {} 

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            return {
                unbroadcast(grad_output, a_shape), 
                unbroadcast(grad_output, b_shape)
            };
        }
    };

    Tensor add(Tensor a, Tensor b) {
        std::vector<int> target_shape = broadcast_shapes(a.get_shape(), b.get_shape());
        Tensor a_ex = a.expand(target_shape);
        Tensor b_ex = b.expand(target_shape);
        Tensor out = Tensor::zeros(target_shape);

        if (a_ex.is_contiguous() && b_ex.is_contiguous() && out.is_contiguous()) {
            kernels::cpu::add_f32(out.numel(), a_ex.data_ptr(), b_ex.data_ptr(), out.data_ptr());
        } else {
            dispatch_binary_op(a_ex, b_ex, out, [](float x, float y) { return x + y; });
        }

        if ((a.requires_grad() || b.requires_grad()) && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<AddBackward>(a.get_shape(), b.get_shape());
            fn -> next_edges.push_back({a.get_grad_fn(), std::make_shared<Tensor>(a)});
            fn -> next_edges.push_back({b.get_grad_fn(), std::make_shared<Tensor>(b)});
            out.set_grad_fn(fn);
        }
        return out;
    }
    
    struct SubBackward : public GradFn {
        // d(a-b)/da = 1, d(a-b)/db = -1
        std::vector<int> a_shape, b_shape;
        SubBackward(std::vector<int> sA, std::vector<int> sB) : a_shape(sA), b_shape(sB) {}
        
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor minus_one = Tensor::ones({1});
            minus_one.at({0}) = -1.0f;
            Tensor neg_grad = axon::mul(grad_output, minus_one);
            return {
                unbroadcast(grad_output, a_shape), 
                unbroadcast(neg_grad, b_shape)
            };
        }
    };

    Tensor sub(Tensor a, Tensor b) {
        std::vector<int> target_shape = broadcast_shapes(a.get_shape(), b.get_shape());
        Tensor a_ex = a.expand(target_shape);
        Tensor b_ex = b.expand(target_shape);
        Tensor out = Tensor::zeros(target_shape);

        if (a_ex.is_contiguous() && b_ex.is_contiguous() && out.is_contiguous()) {
            kernels::cpu::sub_f32(out.numel(), a_ex.data_ptr(), b_ex.data_ptr(), out.data_ptr());
        } else {
            dispatch_binary_op(a_ex, b_ex, out, [](float x, float y) { return x - y; });
        }

        if ((a.requires_grad() || b.requires_grad()) && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<SubBackward>(a.get_shape(), b.get_shape());
            fn -> next_edges.push_back({a.get_grad_fn(), std::make_shared<Tensor>(a)});
            fn -> next_edges.push_back({b.get_grad_fn(), std::make_shared<Tensor>(b)});
            out.set_grad_fn(fn);
        }

        return out;
    }
    
    struct MulBackward : public GradFn {
        Tensor a, b;
        MulBackward(Tensor a_in, Tensor b_in) : a(a_in), b(b_in) {}
        // d(a*b)/da = b, d(a*b)/db = a
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            return {
                unbroadcast(axon::mul(grad_output, b), a.get_shape()), 
                unbroadcast(axon::mul(grad_output, a), b.get_shape())
            };
        }
    };

    Tensor mul(Tensor a, Tensor b) {
        std::vector<int> target_shape = broadcast_shapes(a.get_shape(), b.get_shape());
        Tensor a_ex = a.expand(target_shape);
        Tensor b_ex = b.expand(target_shape);
        Tensor out = Tensor::zeros(target_shape);

        if (a_ex.is_contiguous() && b_ex.is_contiguous() && out.is_contiguous()) {
            kernels::cpu::mul_f32(out.numel(), a_ex.data_ptr(), b_ex.data_ptr(), out.data_ptr());
        } else {
            dispatch_binary_op(a_ex, b_ex, out, [](float x, float y) { return x * y; });
        }

        if ((a.requires_grad() || b.requires_grad()) && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<MulBackward>(a, b);
            fn -> next_edges.push_back({a.get_grad_fn(), std::make_shared<Tensor>(a)});
            fn -> next_edges.push_back({b.get_grad_fn(), std::make_shared<Tensor>(b)});
            out.set_grad_fn(fn);
        }

        return out;
    }
    
    struct DivBackward : public GradFn {
        Tensor a, b;
        DivBackward(Tensor numerator, Tensor denominator) : a(numerator), b(denominator) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor grad_a = axon::div(grad_output, b);
            Tensor b2 = axon::mul(b, b);
            Tensor neg_grad_a_b2 = axon::neg(axon::div(axon::mul(grad_output, a), b2));

            return {
                grad_a, neg_grad_a_b2
            };
        }
    };

    Tensor div(Tensor a, Tensor b) {
        std::vector<int> target_shape = broadcast_shapes(a.get_shape(), b.get_shape());
        Tensor a_ex = a.expand(target_shape);
        Tensor b_ex = b.expand(target_shape);
        Tensor out = Tensor::zeros(target_shape);

        if (a_ex.is_contiguous() && b_ex.is_contiguous() && out.is_contiguous()) {
            kernels::cpu::div_f32(out.numel(), a_ex.data_ptr(), b_ex.data_ptr(), out.data_ptr());
        } else {
            dispatch_binary_op(a_ex, b_ex, out, [](float x, float y) { return x / y; });
        }

        if ((a.requires_grad() || b.requires_grad()) && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<DivBackward>(a, b);
            fn -> next_edges.push_back({a.get_grad_fn(), std::make_shared<Tensor>(a)});
            fn -> next_edges.push_back({b.get_grad_fn(), std::make_shared<Tensor>(b)});
            out.set_grad_fn(fn);
        }

        return out;
    }

    struct NegBackward : public GradFn {
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            return {
                axon::neg(grad_output)
            };
        }
    };

    Tensor neg(Tensor t) {
        Tensor out = Tensor::zeros(t.get_shape());
        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        kernels::cpu::neg_f32(t_c.numel(), t_c.data_ptr(), out.data_ptr());
        
        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<NegBackward>();
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }
        return out;
    }

    struct SqrtBackward : public GradFn {
        Tensor output;
        SqrtBackward(Tensor out) : output(out) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor two = Tensor::ones({1});
            two.at({0}) = 2.0f;
            Tensor denominator = axon::mul(output, two);

            return {
                axon::div(grad_output, denominator)
            };
        }
    };


    Tensor sqrt(Tensor t) {
        Tensor out = Tensor::zeros(t.get_shape());
        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        kernels::cpu::sqrt_f32(t_c.numel(), t_c.data_ptr(), out.data_ptr());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<SqrtBackward>(out); // Save output for backward
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }

        return out;
    }

    struct ExpBackward : public GradFn {
        Tensor output;
        ExpBackward(Tensor out) : output(out) {}
        
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            return {
                axon::mul(grad_output, output)
            };
        }
    };

    Tensor exp(Tensor t) {
        Tensor out = Tensor::zeros(t.get_shape());
        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        kernels::cpu::exp_f32(t_c.numel(), t_c.data_ptr(), out.data_ptr());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<ExpBackward>(out); // Save output for backward
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }

        return out;
    }


    struct TransposeBackward : public GradFn {
        int d0, d1;
        
        TransposeBackward(int dim0, int dim1) : d0(dim0), d1(dim1) {}
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            // Backward of transpose is just transpose again with same dims
            return { axon::transpose(grad_output, d0, d1) };
        }
    };

    Tensor transpose(Tensor t, int dim0, int dim1) {
        std::vector<int> new_shape = t.get_shape();
        std::vector<int> new_stride = t.get_stride();
        std::swap(new_shape[dim0], new_shape[dim1]);
        std::swap(new_stride[dim0], new_stride[dim1]);

        Tensor out = Tensor::from_storage(t.get_storage(), new_shape, new_stride, t.get_offset());
       
        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<TransposeBackward>(dim0, dim1);
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }

        return out;
    }

    size_t get_flat_offset(const std::vector<int>& strides, const std::vector<int>& indices) {
        size_t off = 0;
        for (size_t i = 0; i < strides.size(); i++) {
            off += indices[i] * strides[i];
        }

        return off;
    }

    struct MatMulBackward : public GradFn {
        Tensor a, b;
        MatMulBackward(Tensor a_in, Tensor b_in) : a(a_in), b(b_in) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            int a_rank = a.get_shape().size();
            int b_rank = b.get_shape().size();

            // transpose the last two dimensions
            Tensor a_t = axon::transpose(a, a_rank - 2, a_rank - 1);
            Tensor b_t = axon::transpose(b, b_rank - 2, b_rank - 1);
        
            Tensor grad_a = axon::matmul(grad_output, b_t);
            Tensor grad_b = axon::matmul(a_t, grad_output);

            return {
                unbroadcast(grad_a, a.get_shape()),
                unbroadcast(grad_b, b.get_shape())
            };
        }
    };

    Tensor matmul_impl(Tensor a, Tensor b);

    Tensor matmul(Tensor a, Tensor b) {
        int a_rank = a.get_shape().size();
        int b_rank = b.get_shape().size();
    
        if (a_rank == 1 && b_rank == 1) {
            Tensor a_2d = axon::view(a, {1, a.get_shape()[0]});
            Tensor b_2d = axon::view(b, {b.get_shape()[0], 1});

            Tensor out = matmul_impl(a_2d, b_2d);
            return axon::view(out, {1});
        }
    
        if (a_rank == 2 && b_rank == 1) {
            Tensor b_2d = axon::view(b, {b.get_shape()[0], 1});
            Tensor out = matmul_impl(a, b_2d);
            return axon::view(out, {a.get_shape()[0]});
        }

        if (a_rank == 1 && b_rank == 2) {
            Tensor a_2d = axon::view(a, {1, a.get_shape()[0]});
            Tensor out = matmul_impl(a_2d, b);
            return axon::view(out, {b.get_shape()[1]});
        }

        return matmul_impl(a, b);
    }

    Tensor matmul_impl(Tensor a, Tensor b) {
        int a_rank = a.get_shape().size();
        int b_rank = b.get_shape().size();
   
        if (a_rank < 2 || b_rank < 2) {
            throw std::invalid_argument("[MATMUL_IMPL] Internal Error: Rank of a matrix found < 2");
        }

        int M = a.get_shape()[a_rank - 2];
        int K = a.get_shape()[a_rank - 1];
        int K2 = b.get_shape()[b_rank - 2];
        int N = b.get_shape()[b_rank - 1];

        if (K != K2) {
            throw std::invalid_argument("[MATMUL] Error: Inner shape mismatch");
        }

        std::vector<int> batch_a(a.get_shape().begin(), a.get_shape().end() - 2);
        std::vector<int> batch_b(b.get_shape().begin(), b.get_shape().end() - 2);
   
        std::vector<int> batch_out;
        try {
            batch_out = broadcast_shapes(batch_a, batch_b);
        } catch (...) {
            throw std::invalid_argument("[MATMUL] Batch dimensions mismatch");
        }

        std::vector<int> out_shape = batch_out;
        out_shape.push_back(M);
        out_shape.push_back(N);

        Tensor out = Tensor::zeros(out_shape);

        std::vector<int> shape_a_exp = batch_out;
        shape_a_exp.push_back(M);
        shape_a_exp.push_back(K);

        std::vector<int> shape_b_exp = batch_out;
        shape_b_exp.push_back(K);
        shape_b_exp.push_back(N);

        Tensor a_ex = a.expand(shape_a_exp);
        Tensor b_ex = b.expand(shape_b_exp);

        size_t total_batch = 1;
        for (auto i : batch_out) {
            total_batch *= i;
        } 

        size_t batch_out_size = batch_out.size();

        std::vector<int> a_batch_strides(batch_out_size);
        std::vector<int> b_batch_strides(batch_out_size);
    
        std::vector<int> out_batch_stides(batch_out_size);
        std::vector<int> current_indices(batch_out_size, 0);

        for (size_t i = 0; i < batch_out_size; i++) {
            a_batch_strides[i] = a_ex.get_stride()[i];
            b_batch_strides[i] = b_ex.get_stride()[i];
            out_batch_stides[i] = out.get_stride()[i];

        }

        float* a_ptr_base = a_ex.data_ptr();
        float* b_ptr_base = b_ex.data_ptr();
        float* out_ptr_base = out.data_ptr();

        size_t ax_rank = a_ex.get_shape().size();
        size_t bx_rank = b_ex.get_shape().size();
    

        for (size_t b_idx = 0; b_idx < total_batch; b_idx++) {
            size_t off_a = 0;
            size_t off_b = 0;
            size_t off_o = 0;
       
            for (size_t i = 0; i < batch_out.size(); i++) {
                off_a += current_indices[i] * a_batch_strides[i];
                off_b += current_indices[i] * b_batch_strides[i];
                off_o += current_indices[i] * out_batch_stides[i];
            }

            std::vector<int> stA = {a_ex.get_stride()[ax_rank - 2], a_ex.get_stride()[ax_rank - 1]};
            std::vector<int> stB = {b_ex.get_stride()[bx_rank - 2], b_ex.get_stride()[bx_rank - 1]};

            bool a_contig = (stA[1] == 1 && stA[0] == K);
            bool b_contig = (stB[1] == 1 && stB[0] == N);
        
            std::vector<float> a_buf, b_buf;
            const float* pA = a_ptr_base + off_a;
            const float* pB = b_ptr_base + off_b;

            if (!a_contig) {
                a_buf.resize(M * K);
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < K; j++) {
                        a_buf[i * K + j] = pA[i * stA[0] + j * stA[1]];
                    }
                }
                pA = a_buf.data();
            }

            if (!b_contig) {
                b_buf.resize(K * N);
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < N; j++) {
                        b_buf[i * N + j] = pB[i * stB[0] + j * stB[1]];
                    }
                }
                pB = b_buf.data();
            }

            kernels::cpu::matmul_f32(M, N, K, pA, pB, out_ptr_base + off_o);
            
            for (int i = static_cast<int>(batch_out.size()) - 1; i >= 0; i--) {
                current_indices[i]++;
                if (current_indices[i] < batch_out[i]) {
                    break;
                }
                current_indices[i] = 0;
            }
        }

        if ((a.requires_grad() || b.requires_grad()) && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<MatMulBackward>(a, b);
            fn -> next_edges.push_back({a.get_grad_fn(), std::make_shared<Tensor>(a)});
            fn -> next_edges.push_back({b.get_grad_fn(), std::make_shared<Tensor>(b)});

            out.set_grad_fn(fn);
        }

        return out;
    }

    struct SumBackward : public GradFn {
        std::vector<int> input_shape;
        SumBackward(std::vector<int> shape) : input_shape(shape) {}

        // d(sum(x))/dx = 1 (expanded to shape of x)
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            return {grad_output.expand(input_shape)};
        }
    };

    Tensor sum(Tensor a) {
        Tensor out = Tensor::zeros({1});
        
        if (a.is_contiguous()) {
            // Fast Path: AVX Vectorized Sum
            kernels::cpu::sum_f32(a.numel(), a.data_ptr(), out.data_ptr());
        } else {
            float val = sum_recursive(0, a, 0);
            out.data_ptr()[0] = val;
        }
        
        if (a.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<SumBackward>(a.get_shape());
            fn -> next_edges.push_back({a.get_grad_fn(), std::make_shared<Tensor>(a)});
            out.set_grad_fn(fn);
        }
        
        return out;
    }

    Tensor sum(Tensor t, int dim, bool keepdim) {
        std::vector<int> shape = t.get_shape();
        // Handle negative dims (-1)
        if (dim < 0) {
            dim += shape.size();
        }
    
        if (dim < 0 || dim >= shape.size()) {
            throw std::invalid_argument("[SUM] Error: Invalid dimensions");
        }

        size_t outer = 1;
        for (int i = 0; i < dim; i++) {
            outer *= shape[i];
        }

        size_t target_dim_size = shape[dim];

        size_t inner = 1;

        for (size_t i = dim + 1; i < shape.size(); i++) {
            inner *= shape[i];
        }

        std::vector<int> out_shape;
        for(size_t i = 0; i < shape.size(); ++i) {
            if (i == dim) {
                if (keepdim) {
                    out_shape.push_back(1);
                }                    
            } else {
                out_shape.push_back(shape[i]);
            }
        }
        
        if (out_shape.empty() && !keepdim) {
            out_shape.push_back(1);
        }

        Tensor out = Tensor::zeros(out_shape);

        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        kernels::cpu::sum_dim_f32(outer, target_dim_size, inner, t_c.data_ptr(), out.data_ptr());

        return out;
    }

    // * CUSTOM LAYERS

    struct EmbeddingBackward : public GradFn {
        Tensor weight, indices;
        EmbeddingBackward(Tensor w, Tensor idx) : weight(w), indices(idx) {}
    
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor grad_weight = Tensor::zeros(weight.get_shape());
            Tensor grad_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
            Tensor idx_c = indices.is_contiguous() ? indices : indices.contiguous();

            size_t vocab = weight.get_shape()[0];
            size_t dim = weight.get_shape()[1];
            size_t num_idx = indices.numel();

            kernels::cpu::embedding_backward_f32(vocab, dim, num_idx, grad_out_c.data_ptr(), idx_c.data_ptr(), grad_weight.data_ptr());

            return {
                Tensor::zeros(indices.get_shape()),
                grad_weight
            };
        }
    };

    Tensor embedding(Tensor input, Tensor weight) {
        if (weight.get_shape().size() != 2) {
            throw std::invalid_argument("[EMBEDDING]: Weight must be 2D");
        }

        std::vector<int> out_shape = input.get_shape();

        out_shape.push_back(weight.get_shape()[1]);
        Tensor out = Tensor::zeros(out_shape);

        Tensor input_c = input.is_contiguous() ? input : input.contiguous();

        kernels::cpu::embedding_forward_f32(
            weight.get_shape()[0], weight.get_shape()[1],
            input.numel(), weight.data_ptr(),
            input_c.data_ptr(), out.data_ptr()
        );

        if (weight.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);

            auto fn = std::make_shared<EmbeddingBackward>(weight, input);
            fn -> next_edges.push_back({input.get_grad_fn(), std::make_shared<Tensor>(input)});
            fn -> next_edges.push_back({weight.get_grad_fn(), std::make_shared<Tensor>(weight)});
            out.set_grad_fn(fn);
        }
        
        return out;
    }

    struct LayerNormBackward : public GradFn {
        Tensor input, gamma;
        float eps;

        LayerNormBackward(Tensor in, Tensor g, float e) : input(in), gamma(g), eps(e) {}

        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor grad_input = Tensor::zeros(input.get_shape());
            Tensor grad_gamma = Tensor::zeros(gamma.get_shape());
            Tensor grad_beta = Tensor::zeros(gamma.get_shape());

            size_t cols = input.get_shape().back();
            size_t rows = input.numel() / cols;

            Tensor in_c = input.is_contiguous() ? input : input.contiguous();
            Tensor gam_c = gamma.is_contiguous() ? gamma : gamma.contiguous();
            Tensor g_out_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

            kernels::cpu::layernorm_backward_f32(rows, cols, 
                g_out_c.data_ptr(), in_c.data_ptr(), gam_c.data_ptr(), eps,
                grad_input.data_ptr(), grad_gamma.data_ptr(), grad_beta.data_ptr());
            
            // Output order must match inputs of forward: {input, gamma, beta}
            return {
                grad_input, grad_gamma, grad_beta
            };
        }
    };

    Tensor layer_norm(Tensor input, Tensor gamma, Tensor beta, float eps) {
        
        int dim = input.get_shape().back();
        if (gamma.numel() != dim || beta.numel() != dim) {
            throw std::invalid_argument("[LAYERNORM] Shape mismatch");
        } 

        Tensor out = Tensor::zeros(input.get_shape());

        size_t cols = dim;
        size_t rows = input.numel() / cols;

        Tensor in_c = input.is_contiguous() ? input : input.contiguous();
        Tensor gam_c = gamma.is_contiguous() ? gamma : gamma.contiguous();
        Tensor bet_c = beta.is_contiguous() ? beta : beta.contiguous();

        kernels::cpu::layernorm_forward_f32(rows, cols, in_c.data_ptr(), gam_c.data_ptr(), bet_c.data_ptr(), out.data_ptr(), eps);

        if ((input.requires_grad() || gamma.requires_grad() || beta.requires_grad()) && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<LayerNormBackward>(input, gamma, eps);
            fn -> next_edges.push_back({input.get_grad_fn(), std::make_shared<Tensor>(input)});
            fn -> next_edges.push_back({gamma.get_grad_fn(), std::make_shared<Tensor>(gamma)});
            fn -> next_edges.push_back({beta.get_grad_fn(), std::make_shared<Tensor>(beta)});
            out.set_grad_fn(fn);
        }

        return out;
    }

    struct SoftmaxBackward : public GradFn {
        Tensor output;
        SoftmaxBackward(Tensor out) : output(out) {}
    
        std::vector<Tensor> apply(const Tensor& grad_output) override {
            Tensor grad_input = Tensor::zeros(output.get_shape());
            
            size_t cols = output.get_shape().back();
            size_t rows = output.numel() / cols;
            
            Tensor out_c = output.is_contiguous() ? output : output.contiguous();
            Tensor gout_c = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

            kernels::cpu::softmax_backward_f32(rows, cols, gout_c.data_ptr(), out_c.data_ptr(), grad_input.data_ptr());
            
            return { grad_input };
        }
    };

    Tensor softmax(Tensor t) {
        Tensor out = Tensor::zeros(t.get_shape());
        size_t cols = t.get_shape().back();
        size_t rows = t.numel() / cols;

        Tensor t_c = t.is_contiguous() ? t : t.contiguous();
        kernels::cpu::softmax_f32(rows, cols, t_c.data_ptr(), out.data_ptr());

        if (t.requires_grad() && GradMode::is_enabled()) {
            out.set_requires_grad(true);
            auto fn = std::make_shared<SoftmaxBackward>(out);
            fn -> next_edges.push_back({t.get_grad_fn(), std::make_shared<Tensor>(t)});
            out.set_grad_fn(fn);
        }
        return out;
    }

}
