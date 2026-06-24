#include <autogradpp/autogradpp.hpp>

using namespace autogradpp;

#include <iostream>

int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    double learning_rate = 0.1;
    double stopping_criteria = 1e-8;
    Tensor x_start = Tensor::ones({2});
    Tensor soln = Tensor::zeros({2});
    soln(0) = 2; soln(1) = 5;

    std::cout << "Input Function: z = (x-2)^2 + (y-5)^2" << std::endl;
    std::cout << "Starting Position: " << x_start << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Stopping Criteria: |Δz| < " << stopping_criteria << std::endl;
    std::cout << std::endl;

    double last_z = 0.0;
    int step = 0;

    while (true) {
        // Compute z = (x-2)^2 + (y-5)^2
        auto x = Variable(x_start);
        auto c = Constant(soln);

        auto diff = x - c;
        auto z = matmul((diff * diff), Constant(Tensor::ones({2})));

        if (std::abs(last_z - z.value()) < stopping_criteria && step > 0) { break; }

        // Differentiate and perform gradient descent
        z.backward();

        if ((step+1) % 5 == 0 || step == 0) {
            printf("%3d: x=%.3f, y=%.3f, z=%.4f, ∂z/∂x=%.3f, ∂z/∂y=%.3f\n", step+1, x.value()(0), x.value()(1), (double)z.value(), x.grad()(0), x.grad()(1));
        }

        x_start -= learning_rate * x.grad();
        last_z = z.value();
        step += 1;
    }

    printf("\nMinimum: x=%.3f, y=%.3f, z=%.4f\n", x_start(0), x_start(1), last_z);

    return 0;
}
