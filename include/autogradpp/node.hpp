#pragma once

#include "tensor.hpp"
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

namespace autogradpp {
    class Operand {
    public:
        virtual Tensor forward(std::vector<Tensor>& inputs) = 0;
        virtual std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) = 0;
    };

    struct Node {
    public:
        Tensor value;
        Tensor grad;

        Node(std::unique_ptr<Operand> op, std::vector<std::shared_ptr<Node>> parents, bool requires_grad = true) 
            : _op(std::move(op)), _parents(parents), _requires_grad(requires_grad) {
            std::vector<Tensor> inputs;
            for (auto& parent : parents) {
                inputs.push_back(parent->value);
            }
            value = _op->forward(inputs);
            grad = Tensor::zeros(value.shape());
        }

        void backward() {
            grad = Tensor::ones(value.shape()); 
            auto order = topo_sort();

            for (auto node : order) {
                if (!node->_requires_grad) continue;

                Tensor n_grad = node->grad;

                std::vector<Tensor> inputs;
                for (auto parent : node->_parents) {
                    inputs.push_back(parent->value);
                }
                auto sz = node->_op->backward(inputs, n_grad);

                for (size_t i = 0; i < node->_parents.size(); ++i) {
                    node->_parents[i]->grad += sz[i];
                }
            }
        }
    private:
        std::unique_ptr<Operand> _op;
        std::vector<std::shared_ptr<Node>> _parents;
        bool _requires_grad;

        std::vector<Node*> topo_sort() {
            std::vector<Node*> order;
            std::unordered_set<Node*> visited;

            this->dfs(visited, order);
            std::reverse(order.begin(), order.end());

            return order;
        }

        void dfs(std::unordered_set<Node*>& visited, std::vector<Node*>& order) {
            auto it = visited.find(this);
            if (it != visited.end()) { return; }
            visited.insert(this);

            for (auto parent : _parents) {
                parent->dfs(visited, order);
            }

            order.push_back(this);
        }
    };
}
