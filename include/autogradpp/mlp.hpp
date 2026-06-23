#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include "operands.hpp"
#include <functional>
#include <memory>
#include <stdexcept>

namespace autogradpp {
    using ActivationFn = std::function<std::shared_ptr<Node>(std::shared_ptr<Node>)>;

    namespace activations {
        inline ActivationFn sigmoid = autogradpp::sigmoid;
        inline ActivationFn relu = autogradpp::relu;
        inline ActivationFn tanh = autogradpp::tanh;
        inline ActivationFn hard_tanh = autogradpp::hard_tanh;
    }

    class NeuronLayer {
    public:
        std::shared_ptr<Node> weight;
        std::shared_ptr<Node> bias;
        ActivationFn activation_fn;

        NeuronLayer(size_t num_inputs, size_t num_outputs, ActivationFn fn = relu)
            : weight(var(Tensor::rand(-1.0, 1.0, {num_outputs, num_inputs}))), 
              bias(var(Tensor::zeros({num_outputs}))),
              activation_fn(fn) {
        }

        std::shared_ptr<Node> forward(const Tensor& x) {
            return forward(input(x));
        }

        std::shared_ptr<Node> forward(std::shared_ptr<Node> x) {
            /*if (x->value.shape() != weight->value.shape()) {
                throw std::runtime_error("forward: input is not the same shape as weight tensor.");
            }*/

            auto pre_activation = add(matmul(weight, x), bias);
            //std::cout << "pre_activation for W=" << weight->value << ": " << pre_activation->value << std::endl;
            return activation_fn(pre_activation);
        }
    };

    class NeuralNetwork {
    public:
        NeuralNetwork(std::vector<size_t> layer_sizes, std::vector<ActivationFn> activation_fns = {}) {
            if (layer_sizes.size() < 2) {
                throw std::runtime_error("Neural network requires at least two layers.");
            }

            if (layer_sizes.size() != activation_fns.size() + 1 && !activation_fns.empty()) {
                throw std::runtime_error("Wrong number of activation functions provided. There should be one per non-input layer.");
            }

            size_t in_dim = layer_sizes[0];
            for (size_t i = 1; i < layer_sizes.size(); ++i) {
                size_t out_dim = layer_sizes[i]; 
                if (activation_fns.empty()) {
                    _layers.push_back(std::make_unique<NeuronLayer>(in_dim, out_dim));
                }
                else {
                    _layers.push_back(std::make_unique<NeuronLayer>(in_dim, out_dim, activation_fns[i-1]));
                }
                in_dim = out_dim;
            }
        }

        std::shared_ptr<Node> forward(Tensor x) {
            return forward(var(x));
        }

        std::shared_ptr<Node> forward(std::shared_ptr<Node> x) {
            std::shared_ptr<Node> current = x;
            for (auto& n : _layers) {
                current = n->forward(current);
            }

            if (current->value.size() == 1) { 
                current->value = current->value.reshape({}); 
                current->grad = current->grad.reshape({}); 
            }
            return current;
        }       

        std::vector<std::shared_ptr<Node>> parameters() {
            std::vector<std::shared_ptr<Node>> ps;
            for (auto& layer : _layers) {
                ps.push_back(layer->weight);
                ps.push_back(layer->bias);
            }
            return ps;
        }
    private:
        std::vector<std::unique_ptr<NeuronLayer>> _layers;
    };
}
