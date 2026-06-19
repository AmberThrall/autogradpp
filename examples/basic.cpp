#include <autogradpp/autogradpp.hpp>

using namespace autogradpp;

#include <iostream>

int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    Tensor<float> x = Tensor<float>::rand(-1, 1, {2, 2});
    x(0, 0) = 2;
    x(1, 1) = 3;
    std::cout << x << std::endl;

    Tensorf eye = Tensorf::eye(3);
    std::cout << eye << std::endl;


    return 0;
}
