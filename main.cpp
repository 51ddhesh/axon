#include "./include/Tensor.hpp"

int main() {
    std::vector<int> v = {1, 2, 3, 4, 5};
    Tensor<int> t(v);
    t.print_tensor();
    std::cout << "Tensor check!" << std::endl;
    print(t);
    return 0;
}

