# autogradpp

autogradpp is a C++, header only library for computational graphs and automatic differentiation.
It makes use of 0-,1- and 2-tensors for all operations. 
Also included is a helper class for constructing multilayer perceptron networks.

# Example

```cpp
int main() {
    double learning_rate = 0.1;
    size_t num_epochs = 10000;

    std::vector<std::pair<Tensor, double>> dataset = {
        {Tensor::vector({0, 0}), 0},
        {Tensor::vector({0, 1}), 1},
        {Tensor::vector({1, 0}), 1},
        {Tensor::vector({1, 1}), 0},
    };

    // Create a multilayer perceptron with one hidden layer. Use tanh as the activation function.
    NeuralNetwork network(2, {
        {4, activations::tanh}, 
        {1, activations::tanh}
    });

    // Perform batch gradient descent `num_epochs` times
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Construct the graph
        auto total_loss = input(0.0);
        for (auto& [x_val, y_val] : dataset) {
            auto y_pred = network.forward(x_val);
            auto y_true = constant(y_val);
            auto loss = sub(y_true, y_pred);

            total_loss = add(total_loss, mul(loss, loss));
        }
            
        // Gradient descent
        total_loss->backward();
        for (auto param : network.parameters()) {
            param->value -= learning_rate * param->grad;
            param->grad = Tensor::zeros(param->grad.shape());
        }

        if ((epoch+1) % 1000 == 0 || epoch == 0) {
            std::cout << "  epoch " << std::setw(5) << epoch + 1 << " loss=" << total_loss->value / 4.0 << std::endl;
        }
    }

    // Test the trained model
    double mse = 0;
    for (auto& [x_val, y_val] : dataset) {
        auto y_pred = network.forward(x_val);
        auto y_true = constant(y_val);

        std::cout << "  x=" << x_val  << " -> pred=" << y_pred->value << " (target=" << y_true->value << ")" << std::endl;

        double err = y_pred->value - y_true->value;
        mse += err * err;
    }
    std::cout << std::endl << "Mean-squared Error: " << mse / 4 << std::endl;

    return 0;
}
```

