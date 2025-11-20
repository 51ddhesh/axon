#include "include/Tensor.hpp"
#include "include/Activations.hpp"
#include "include/Linear.hpp"
#include "include/Sequential.hpp"
#include "include/LossFunctions.hpp"
#include "include/TensorMath.hpp"

int main() {
    // Tensor t = Tensor::zeros(2, 2);
    // print(t);

    // Tensor a({{1, 2}, {3, 4}});
    // Tensor b({{4, 2}, {1, 4}});

    // print(a + b);
    // print(a - b);

    // Tensor c(1, 1);
    // print(c);
    
    // c = c - 1;
    // print(c);

    // c = c * 5;
    // c += 10;
    // print(c);

    // Tensor A(1, 1);
    // Tensor B(1, 1);
    // A = A + 1;
    // B = B + 2;
    
    // Tensor C = A + B;

    // print(C);
    // C += A;
    // print(C);

    // C += 5;
    // print(C);

    // print(Tensor(4, 4, 0.5));

    // t = Tensor::ones(2, 2);
    // print(t);

    // t = Tensor::zeros(2, 2);
    // print(t);

    // std::cout << "Matrix B:" << std::endl;
    // print(b);

    // Tensor a_b = matmul(a, b);
    // std::cout << "A * B" << std::endl;
    // print(a_b);

    // Tensor X = Tensor::randn(3, 4, {2, 4});
    // print(X);

    // print(X.T());

    // Tensor Y = Tensor::randint(3, 4, {100, 120});
    // print(Y);

    // Tensor softmax_test = Tensor::randn(3, 4);
    // print(softmax_test);

    // Tensor softmaxed_ = axon_activation::softmax(softmax_test);
    // print(softmaxed_);

    // // Tensor with batch size = 2, features = 10
    // Tensor a1 = Tensor::randn(2, 10);
    // std::cout << "Input Tensor:\n";
    // print(a1);

    // // Create a torch.nn.Linear -esque object or a tf.layers.Dense
    // // 10 features, 5 neurons (outputs), ReLU activation
    // Linear layer(10, 5, axon_activation::relu);

    // Tensor output = layer.linear(a1);
    // std::cout << "Running the forward pass:\n";
    // print(output);

    // // Verify the shape of the output
    // std::cout << "Output shape: (" << output.rows() << ", " << output.cols() << ")\n";
    
    // // Create an input Tensor, batch_size = 4, features = 20
    // Tensor in = Tensor::randn(4, 16);
    // std::cout << "Input Tensor\n";
    // print(in);
    
    // Sequential layers;
    // // Add the first layer: input = 20, neurons = 16, ReLU
    // layers.add(Linear(16, 16, axon_activation::relu));
    // // Add the second layer: input = 16, neurons = 8, ReLU
    // layers.add(Linear(16, 8, axon_activation::relu));
    // // Add the third layer: input = 8, output = 3, softmax
    // layers.add(Linear(8, 3, axon_activation::softmax));
    
    // output = layers.linear(in); // or layers.forward(in)
    // std::cout << "Output shape: (" << output.rows() << ", " << output.cols() << ")\n";
    // std::cout << "\nOutput Tensor:\n";
    // print(output);


    // // Testing the Mean Squared Error
    // std::cout << "MSE Test:\n";
    // Tensor y_pred_mse = Tensor::randn(1, 10);
    // Tensor y_true_mse = Tensor::randn(1, 10);

    // double mse = axon_loss::mse(y_pred_mse, y_true_mse);
    // std::cout << "Y Predicted:\n";
    // print(y_pred_mse);
    // std::cout << "Y True:\n";
    // print(y_true_mse);
    // std::cout << "Loss: " << mse << std::endl;


    // // Testing the Categorical Cross-Entropy Loss
    // std::cout << "CCE Test:\n";
    // Tensor y_pred_cce({
    //     {0.1, 0.8, 0.1},
    //     {0.9, 0.05, 0.05}
    // });

    // Tensor y_true_cce({
    //     {0.0, 1.0, 0.0},
    //     {1.0, 0.0, 0.0}
    // });

    // double cce = axon_loss::cce(y_pred_cce, y_true_cce);
    // std::cout << "Y Predicted:\n";
    // print(y_pred_cce);
    // std::cout << "Y True Labels:\n";
    // print(y_true_cce);
    // std::cout << "Loss: " << cce << std::endl;

    // Tensor t2({10, 10, 10});
    // Tensor t1 = Tensor::randn(3, 3);
    // print(t1);
    // t1 *= 10;
    // print(t1);
    // print(t1 / t2);

    // Tensor t1 = Tensor::zeros(3, 3);
    // print(t1);
    // std::cout << "----------------------------------------\n";
    // Tensor t2 = Tensor::column({1, 2, 3});
    // print(t2);
    // std::cout << "----------------------------------------\n";
    // print(t1 + t2);

    // Tensor a = Tensor::randn(2, 2);
    // Tensor b = Tensor::randn(1, 1);

    // print(a); print(b);
    // b(0) = 1.000;
    // Tensor is_g = axon::math::gt(b, a);
    // print(is_g);

    // a = Tensor::randn(2, 2);
    // b = Tensor::randn(2, 2);
    
    // a(1) = 1.0;
    // b(1) = 1.0;
    
    // print(a); print(b);
    // Tensor is_eq = axon::math::eq(a, b);
    // print(is_eq);

    // Tensor a = {
    //     {2, 1},
    //     {0, 3}
    // };

    // print(a);

    // std::cout << "---------------------------\n";
    
    // auto b = axon::math::log2(a);
    
    // print(b);
    
    // print(axon::math::pow(2, b));
    
    // print(axon::math::pow(a, Tensor::zeros(1, 1)));
    
    // Tensor a = Tensor::randn(2, 2);
    // print(a);
    // std::cout << "---------------------------\n";
    // Tensor b = Tensor::randn(2, 2);
    // print(b);
    // std::cout << "---------------------------\n";
    
    // print(axon::math::maximum(a, b));
    // std::cout << "---------------------------\n";
    // print(axon::math::minimum(a, b));
    
    // print(a.sum()); 
    // print(b.sum());
    
    return 0;
}
