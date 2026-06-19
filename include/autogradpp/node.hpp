#pragma once

#include "tensor.hpp"
#include <memory>

namespace autogradpp {
    class Operand {
    public:
        virtual Tensord forward(std::vector<Tensord>& inputs) = 0;
    };

    struct Node {
    public:
        Tensord value;

        Node(std::unique_ptr<Operand> op, std::vector<std::shared_ptr<Node>> parents, bool requires_grad = true) 
            : _op(std::move(op)), _parents(parents), _requires_grad(requires_grad) {
            std::vector<Tensord> inputs;
            for (auto& parent : parents) {
                inputs.push_back(parent->value);
            }
            value = _op->forward(inputs);
        }
    private:
        std::unique_ptr<Operand> _op;
        std::vector<std::shared_ptr<Node>> _parents;
        bool _requires_grad;
    };
}
