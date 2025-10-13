#include "include/Tensor.hpp"

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

    // Tensor a(1, 1);
    // Tensor b(1, 1);
    // a = a + 1;
    // b = b + 2;
    // Tensor c = a + b;
    // print(c);
    // c += a;
    // print(c);

    // c += 5;
    // print(c);

    // print(Tensor(4, 4, 0.5));

    Tensor t = Tensor::randn(2, 2, 0, 5);
    print(t);

    t = Tensor::ones(2, 2);
    print(t);

    t = Tensor::zeros(2, 2);
    print(t);

    

    return 0;
}
