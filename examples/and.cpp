#include <autogradpp/autogradpp.hpp>

using namespace autogradpp;

#include <iomanip>
#include <iostream>

int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    double learning_rate = 0.1;
    size_t num_epochs = 10000;

    std::vector<std::pair<Tensor, double>> dataset = {
        {Tensor::vector({0, 0}), 0},
        {Tensor::vector({0, 1}), 0},
        {Tensor::vector({1, 0}), 0},
        {Tensor::vector({1, 1}), 1},
    };

    std::cout << "Desired Function: AND" << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Number of Epochs: " << num_epochs << std::endl;
    std::cout << std::endl;

    NeuralNetwork neuron(2, {{1, activations::sigmoid}});

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        for (auto& [x_val, y_val] : dataset) {
            auto y_pred = neuron.forward(x_val);
            auto y_true = Constant(y_val);
            auto loss = (y_pred - y_true) * (y_pred - y_true);

            total_loss += loss.value();
            loss.backward();

            // Gradient descent
            for (auto param : neuron.parameters()) {
                param.value() -= learning_rate * param.grad();
                param.grad() = Tensor::zeros(param.grad().shape());
            }
        }

        if ((epoch+1) % 1000 == 0 || epoch == 0) {
            std::cout << "  epoch " << std::setw(5) << epoch + 1 << " loss=" << total_loss / 4 << std::endl;
        }
    }

    // Test the trained model
    std::cout << std::endl << "After Training:" << std::endl;
    double mse = 0;
    for (auto& [x_val, y_true] : dataset) {
        auto y_pred = neuron.forward(x_val);

        std::cout << "  x=" << x_val  << " -> pred=" << y_pred.value() << " (target=" << y_true << ")" << std::endl;

        double err = y_pred.value() - y_true;
        mse += err * err;
    }
    std::cout << std::endl << "Mean-squared Error: " << mse << std::endl;

    return 0;
}
