#include <autogradpp/autogradpp.hpp>

using namespace autogradpp;

#include <iomanip>
#include <iostream>

std::pair<Tensor, double> build_datum(double a, double b, double y) {
    Tensor t({2});
    t(0) = a; t(1) = b;
    return {t, y};
}

std::shared_ptr<Node> mse(std::shared_ptr<Node> y_pred, std::shared_ptr<Node> y_true) {
    auto diff = sub(y_pred, y_true);
    auto diff_sq = mul(diff, diff);
    auto loss = matmul(diff_sq, constant(Tensor::ones(diff_sq->value.shape())));
    auto n = constant(Tensor::scalar(diff_sq->value.size()));
    return div(loss, n);
}

int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    double learning_rate = 0.1;
    size_t num_epochs = 10000;

    std::vector<std::pair<Tensor, double>> dataset = {
        build_datum(0, 0, 0),
        build_datum(0, 1, 1),
        build_datum(1, 0, 1),
        build_datum(1, 1, 0),
    };

    std::cout << "Desired Function: XOR" << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Number of Epochs: " << num_epochs << std::endl;
    std::cout << std::endl;

    NeuralNetwork network({2, 4, 1}, {activations::tanh, activations::tanh});

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        double total_loss = 0.0;
        for (auto& [x_val, y_val] : dataset) {
            auto y_pred = network.forward(x_val);
            auto y_true = constant(y_val);
            auto loss = mse(y_pred, y_true);

            loss->backward();
            total_loss += loss->value;

            // Gradient descent
            for (auto param : network.parameters()) {
                param->value -= learning_rate * param->grad;
                param->grad = Tensor::zeros(param->grad.shape());
            }
        }

        if ((epoch+1) % 1000 == 0 || epoch == 0) {
            std::cout << "  epoch " << std::setw(5) << epoch + 1 << " loss=" << total_loss / 4 << std::endl;
        }
    }

    // Test the trained model
    std::cout << std::endl << "After Training:" << std::endl;
    double mse = 0;
    for (auto& [x_val, y_val] : dataset) {
        auto y_pred = network.forward(x_val);
        auto y_true = constant(y_val);

        std::cout << "  x=" << x_val  << " -> pred=" << y_pred->value << " (target=" << y_true->value << ")" << std::endl;

        double err = y_pred->value - y_true->value;
        mse += err * err;
    }
    std::cout << std::endl << "Mean-squared Error: " << mse << std::endl;


    return 0;
}
