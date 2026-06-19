#include <autogradpp/autogradpp.hpp>

using namespace autogradpp;

#include <iostream>

int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    Tensorf eye = Tensorf::eye(3);
    Tensorf ones = Tensorf::ones({3, 3});

    std::cout << eye / 3 - 2 * ones << std::endl;

    Tensorf x = Tensorf::randn({2});
    Tensord c = 2 * Tensord::ones({2});

    std::cout << x << " -> " << c * x << std::endl;

    Tensorf neg = Tensorf::ones({2, 2, 2}) * -1;
    Tensorf pos = neg;
    pos.map([](float x) { return std::abs(x); });
    std::cout << neg << " -> " << pos << std::endl;

    Tensorf scalar = Tensorf::scalar(3.14);
    std::cout << scalar << std::endl;

    // Test matrix multiplication
    Tensorf a({2, 2});
    a(0, 0) = 1;
    a(0, 1) = 2;
    a(1, 0) = 3;
    a(1, 1) = 4;
    x = Tensorf::ones({2});

    std::cout << "matmul(" << a << "," << x << ") = " << matmul(a, x) << std::endl;
    std::cout << "matmul(" << x << "," << a << ") = " << matmul(x, a) << std::endl;
    std::cout << "matmul(" << a << "," << a << ") = " << matmul(a, a) << std::endl;
    std::cout << "matmul(" << x << "," << x << ") = " << matmul(x, x) << std::endl;

    return 0;
}
