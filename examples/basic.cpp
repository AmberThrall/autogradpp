#include <autogradpp/autogradpp.hpp>

using namespace autogradpp;

#include <iostream>

int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    auto x = input(Tensor::ones({5}));
    auto y = constant(Tensor::zeros({3}));
    auto w = var(Tensor::randn({5, 3}));
    auto b = var(Tensor::randn({3}));

    auto z = add(matmul(x, w), b);

    std::cout << z->value << std::endl;

    return 0;
}
