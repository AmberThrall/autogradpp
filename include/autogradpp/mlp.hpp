#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include "operands.hpp"
#include <memory>
#include <stdexcept>

namespace autogradpp {
    template <class T=ReLUOp>
    class Neuron {
    public:
        std::shared_ptr<Node> weight;
        std::shared_ptr<Node> bias;

        Neuron(size_t num_inputs) : weight(var(Tensor::rand(-1.0, 1.0, {num_inputs}))), bias(var(Tensor::scalar(0.0))) {}

        std::shared_ptr<Node> forward(const Tensor& x) {
            return forward(input(x));
        }

        std::shared_ptr<Node> forward(std::shared_ptr<Node> x) {
            if (x->value.shape() != weight->value.shape()) {
                throw std::runtime_error("forward: input is not the same shape as weight tensor.");
            }

            auto pre_activation = add(matmul(weight, x), bias);
            auto op = std::make_unique<T>();
            return std::make_shared<Node>(Node(std::move(op), {pre_activation}));
        }
    };
}
