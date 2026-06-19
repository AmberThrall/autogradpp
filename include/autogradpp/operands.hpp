#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include <memory>

namespace autogradpp {
    class InputOp : public Operand {
    public:
        InputOp(Tensor val) : _val(val) {}

        Tensor forward(std::vector<Tensor>& inputs) {
            return _val;
        }
    private:
        Tensor _val;
    };

    inline std::shared_ptr<Node> input(Tensor val) {
        auto op = std::make_unique<InputOp>(val);
        return std::make_shared<Node>(Node(std::move(op), {}, false));
    }

    inline std::shared_ptr<Node> var(Tensor val) {
        auto op = std::make_unique<InputOp>(val);
        return std::make_shared<Node>(Node(std::move(op), {}));
    }

    inline std::shared_ptr<Node> constant(Tensor val) {
        auto op = std::make_unique<InputOp>(val);
        return std::make_shared<Node>(Node(std::move(op), {}));
    }

    class MulOp : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] * inputs[1];
        }
    };

    inline std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<MulOp>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }

    class AddOp : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] + inputs[1];
        }
    };

    inline std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<AddOp>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }


    class MatMulOp : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return matmul(inputs[0], inputs[1]);
        }
    };

    inline std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<MatMulOp>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }
}
