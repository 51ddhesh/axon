#include "./include/tensor.hpp"


int main() {
    Tensor<float> t1({1.0, 2.0, 3.0, 4.0});
    Tensor<int> t2 = {1, 2, 3, 4, 5};   
    std::vector<int> v = {1, 2, 3, 4, 5};
    Tensor<int> t3(v);
    
    std::cout << "Tensor t1: ";
    print(t1);
    std::cout << "Tensor t2: ";
    t2.print_tensor();
    std::cout << "Tensor t3: ";
    print(t3);

    auto i = t2 + t3;
    print(i);

    i = t2 * t3;
    print(i);
   
    i = t2 - t3;
    print(i);

    t1 = {1.0, 2.0, 3.0, 4.0, 5.0};

    axon::i64 dot_prod = dot(t1, t2);
    std::cout << "The dot between t1 and t2 is: " << dot_prod << std::endl;

    return 0;
}
