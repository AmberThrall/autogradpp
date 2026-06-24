#include <autogradpp/autogradpp.hpp>
#include "autogradpp/mlp.hpp"
#include "indicators/indicators.hpp"
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>

using namespace autogradpp;
using namespace indicators;

std::string format_number(double x, size_t width = 0) {
    std::ostringstream oss;
    oss << std::setw(width) << x;
    return oss.str();
}

class MNISTDataset {
public:
    MNISTDataset(const std::string& images_path, const std::string& labels_path) {
        read_images(images_path);
        read_labels(labels_path);
    }

    size_t num_images() const { return _num_images; }
    size_t image_size() const { return _image_size; }

    std::pair<Tensor, int> sample(size_t id) {
        return {_images[id], _labels[id]};
    }
private:
    std::vector<Tensor> _images;
    std::vector<int> _labels;
    size_t _num_images;
    size_t _image_size;
    
    void read_images(const std::string& path) {
        std::ifstream fp(path, std::ios::binary);
        if (!fp) throw std::runtime_error("Cannot open: " + path);

        uint32_t magic, num_images, rows, cols;
        fp.read(reinterpret_cast<char*>(&magic), 4);
        fp.read(reinterpret_cast<char*>(&num_images), 4);
        fp.read(reinterpret_cast<char*>(&rows), 4);
        fp.read(reinterpret_cast<char*>(&cols), 4);

        magic = swap_endian(magic);
        _num_images = swap_endian(num_images);
        rows = swap_endian(rows);
        cols = swap_endian(cols);

        if (magic != 2051) { throw std::runtime_error("Invalid magic number (expected 2051)."); }

        _image_size = rows * cols;
        for (size_t i = 0; i < _num_images; ++i) {
            Tensor t({_image_size});
            for (size_t j = 0; j < _image_size; ++j) {
                uint8_t pixel;
                fp.read(reinterpret_cast<char*>(&pixel), 1);
                t(j) = pixel / 255.0;
            }
            _images.push_back(t);
        }
    }

    void read_labels(const std::string& path) {
        std::ifstream fp(path, std::ios::binary);
        if (!fp) throw std::runtime_error("Cannot open: " + path);

        uint32_t magic, num_labels;
        fp.read(reinterpret_cast<char*>(&magic), 4);
        fp.read(reinterpret_cast<char*>(&num_labels), 4);

        magic = swap_endian(magic);
        num_labels = swap_endian(num_labels);

        if (magic != 2049) { throw std::runtime_error("Invalid magic number (expected 2049)."); }
        if (num_labels != _num_images) { throw std::runtime_error("Number of labels doesn't match number of images."); }

        for (size_t i = 0; i < _num_images; ++i) {
            uint8_t label;
            fp.read(reinterpret_cast<char*>(&label), 1);
            _labels.push_back(label);
        }

    }

    uint32_t swap_endian(uint32_t val) {
        return ((val & 0xFF000000) >> 24) |
               ((val & 0x00FF0000) >> 8)  |
               ((val & 0x0000FF00) << 8)  |
               ((val & 0x000000FF) << 24);
    }
};

class CrossEntropyLoss : public Operand {
public:
    Tensor forward(std::vector<Tensor>& inputs) {
        auto& y_hat = inputs[0];
        auto& y = inputs[1];
        double loss = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            double p = std::max(y_hat(i), 1e-7);
            loss -= y(i) * std::log(p);
        }
        return Tensor::scalar(loss);
    }

    std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
        return { (inputs[0] - inputs[1]) * grad, Tensor::zeros(inputs[1].shape()) };
    }        
};

std::shared_ptr<Node> cross_entropy(std::shared_ptr<Node> y_hat, std::shared_ptr<Node> y_true) {
    auto op = std::make_unique<CrossEntropyLoss>();
    return std::make_shared<Node>(Node(std::move(op), {y_hat, y_true}));
}


int main() {
    std::cout << "autogradpp version: " << autogradpp::version() << std::endl;

    std::cout << "Loading training set..." << std::flush;
    MNISTDataset dataset("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
    std::cout << "Done" << std::endl;

    double learning_rate = 0.01;
    size_t num_epochs = 20;
    size_t batch_size = 16;
    size_t num_batches = std::ceil((float)dataset.num_images() / (float)batch_size);

    std::cout << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Number of Epochs: " << num_epochs << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Number of Batches: " << num_batches << std::endl;
    std::cout << std::endl;

    // Create a multilayer perceptron with two hidden layers.
    NeuralNetwork network(dataset.image_size(), {
        {128, activations::relu}, 
        {64, activations::relu}, 
        {10, activations::linear}
    });

    std::cout << "== Training ==" << std::endl;
    ProgressBar epoch_bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"■"},
        option::Lead{"■"},
        option::Remainder{" "},
        option::End{" ]"},
        option::MaxProgress{num_epochs},
        option::ForegroundColor{Color::yellow},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::PrefixText{"Epoch "},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };
    ProgressBar batch_bar{
        option::BarWidth{46},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{" ]"},
        option::MaxProgress{num_batches},
        option::ForegroundColor{Color::white},
        option::ShowPercentage{true},
        option::ShowElapsedTime{false},
        option::ShowRemainingTime{false},
        option::PrefixText{"Batch "},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
    };
    DynamicProgress<ProgressBar> bars(epoch_bar, batch_bar);

    // We want to shuffle the order of images each epoch
    std::vector<size_t> image_order;
    for (size_t i = 0; i < dataset.num_images(); ++i) { image_order.push_back(i); }
    std::random_device rng_device;
    std::mt19937 rng(rng_device());

    // Perform mini-batch gradient descent `num_epochs` times
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        epoch_bar.set_option(option::PrefixText { "Epoch " + format_number(epoch+1, 2) + "/" + std::to_string(num_epochs) + " " });
        batch_bar.set_progress(0);

        for (size_t batch = 0; batch < num_batches; ++batch) {
            batch_bar.set_option(option::PrefixText { "Batch " + format_number(batch+1, 4) + "/" + std::to_string(num_batches) + " " });

            // Construct the graph
            double actual_batch_size = 0;
            auto total_loss = input(0.0);
            for (size_t i = 0; i < batch_size; ++i) {
                actual_batch_size += 1;
                size_t id = batch * batch_size + i;
                if (id >= dataset.num_images()) { break; }

                auto datum = dataset.sample(image_order[id]);
                auto logits = network.forward(datum.first);
                auto y_pred = softmax(logits);

                Tensor label_onehot = Tensor::zeros({10});
                label_onehot[datum.second] = 1;
                auto y_true = constant(label_onehot);

                auto loss = cross_entropy(y_pred, y_true);
                total_loss = add(total_loss, loss);
            }
            total_loss = div(total_loss, constant(Tensor::scalar(actual_batch_size)));
                
            // Gradient descent
            total_loss->backward();
            for (auto param : network.parameters()) {
                double norm = 0.0;
                for (size_t i = 0; i < param->grad.size(); ++i) { 
                    norm += param->grad[i] * param->grad[i];
                }

                param->value -= learning_rate * param->grad;
                param->grad = Tensor::zeros(param->grad.shape());
            }
            batch_bar.set_option(option::PostfixText { " loss=" + std::to_string(total_loss->value) + " " });

            bars[1].tick();
        }
        bars[1].mark_as_completed();

        std::shuffle(image_order.begin(), image_order.end(), rng);
        bars[0].tick();
    }

    epoch_bar.set_option(option::PostfixText { std::to_string(num_epochs) + "/" + std::to_string(num_epochs) });
    epoch_bar.mark_as_completed();

    std::cout << std::endl;
    std::cout << "Loading testing set..." << std::flush;
    MNISTDataset testing("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte");
    std::cout << "Done" << std::endl;

    std::cout << std::endl;
    std::cout << "== Testing ==" << std::endl;
    ProgressBar testing_bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"■"},
        option::Lead{"■"},
        option::Remainder{" "},
        option::End{" ]"},
        option::MaxProgress{testing.num_images()},
        option::ForegroundColor{Color::yellow},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    // Test the trained model
    double score = 0.0;
    #pragma omp parallel for
    for (size_t id = 0; id < testing.num_images(); ++id) {
        auto datum = testing.sample(id);
        auto y_pred = network.forward(datum.first);

        int guess = 0;
        double guess_confidence = -10000;
        for (size_t i = 0; i < 10; ++i) {
            if (y_pred->value(i) > guess_confidence) {
                guess = i;
                guess_confidence = y_pred->value(i);
            }
        }

        #pragma omp critical 
        {
            if (guess == datum.second) { score += 1.0; }
            testing_bar.tick();
        }
    }

    std::cout << std::endl;
    std::cout << "Accuracy: " << (score / double(testing.num_images())) * 100.0 << "%" << std::endl;

    return 0;
}
