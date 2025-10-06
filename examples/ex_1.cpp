#include "../include/Tensor.hpp"

int main() {
    Tensor t1({1, 2, 3, 4, 5});
    std::vector<double> d = {10, 11, 12, 13, 14};
    Tensor t2 = d;

    t1 = t1 + t2;

    print(t1);

    t1 = t1 + 6.10;

    print(t1);

    return 0;
}
