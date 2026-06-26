#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include "operands.hpp"
#include <functional>
#include <memory>
#include <stdexcept>

namespace autogradpp {
    enum class Init { He, Xavier, Uniform };
    struct ActivationFn {
        std::function<Variable(Variable&)> fn;
        Init init;
    };

    namespace activations {
        inline ActivationFn linear = { autogradpp::linear, Init::Xavier };
        inline ActivationFn sigmoid = { autogradpp::sigmoid, Init::Xavier };
        inline ActivationFn relu = { autogradpp::relu, Init::He };
        inline ActivationFn tanh = { autogradpp::tanh, Init::Xavier };
        inline ActivationFn hard_tanh = { autogradpp::hard_tanh, Init::Xavier };
    }

    class NeuronLayer {
    public:
        Variable weight;
        Variable bias;
        ActivationFn activation_fn;

        NeuronLayer(size_t num_inputs, size_t num_outputs, ActivationFn fn = activations::relu)
            : weight(Variable(0)),
              bias(Variable(Tensor::zeros({num_outputs}))),
              activation_fn(fn) {
            switch (fn.init) {
                case Init::He:
                    weight = Variable(Tensor::randn({num_outputs, num_inputs}, 0.0, std::sqrt(2.0 / num_inputs)));
                    break;
                case Init::Xavier:
                    weight = Variable(Tensor::randn({num_outputs, num_inputs}, 0.0, std::sqrt(1.0 / num_inputs)));
                    break;
                case Init::Uniform:
                    weight = Variable(Tensor::rand({num_outputs, num_inputs}, -1.0, 1.0));
                    break;
            }
        }

        Variable forward(const Tensor& x) {
            return forward(Variable(x));
        }

        Variable forward(Variable x) {
            auto pre_activation = matmul(weight, x) + bias;
            return activation_fn.fn(pre_activation);
        }
    };

    class NeuralNetwork {
    public:
        NeuralNetwork(size_t input_size, std::vector<size_t> layers) {
            if (layers.size() < 1) {
                throw std::runtime_error("Neural network requires at least two layers.");
            }

            size_t in_dim = input_size;
            for (size_t i = 0; i < layers.size(); ++i) {
                size_t out_dim = layers[i]; 
                _layers.push_back(std::make_unique<NeuronLayer>(in_dim, out_dim, activations::relu));
                in_dim = out_dim;
            }
        }

        NeuralNetwork(size_t input_size, std::vector<std::pair<size_t, ActivationFn>> layers) {
            if (layers.size() < 1) {
                throw std::runtime_error("Neural network requires at least two layers.");
            }

            size_t in_dim = input_size;
            for (size_t i = 0; i < layers.size(); ++i) {
                size_t out_dim = layers[i].first; 
                _layers.push_back(std::make_unique<NeuronLayer>(in_dim, out_dim, layers[i].second));
                in_dim = out_dim;
            }
        }

        Variable forward(Tensor x) {
            return forward(Variable(x));
        }

        Variable forward(Variable x) {
            Variable current = x;
            for (auto& n : _layers) {
                current = n->forward(current);
            }

            if (current.value().size() == 1) { 
                current.value() = current.value().reshape({}); 
                current.grad() = current.grad().reshape({}); 
            }
            return current;
        }       

        std::vector<Variable*> parameters() {
            std::vector<Variable*> ps;
            for (auto& layer : _layers) {
                ps.push_back(&layer->weight);
                ps.push_back(&layer->bias);
            }
            return ps;
        }
    private:
        std::vector<std::unique_ptr<NeuronLayer>> _layers;
    };
}
