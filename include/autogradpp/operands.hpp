#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include <cmath>
#include <memory>

namespace autogradpp {
    class InputOp : public Operand {
    public:
        InputOp(Tensor val) : _val(val) {}

        Tensor forward(std::vector<Tensor>& inputs) {
            return _val;
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {};    
        }
    private:
        Tensor _val;
    };

    inline std::shared_ptr<Node> input(Tensor val) {
        auto op = std::make_unique<InputOp>(val);
        return std::make_shared<Node>(Node(std::move(op), {}, false));
    }
    inline std::shared_ptr<Node> input(double val) { return input(Tensor::scalar(val)); }

    inline std::shared_ptr<Node> var(Tensor val) {
        auto op = std::make_unique<InputOp>(val);
        return std::make_shared<Node>(Node(std::move(op), {}));
    }
    inline std::shared_ptr<Node> var(double val) { return var(Tensor::scalar(val)); }

    inline std::shared_ptr<Node> constant(Tensor val) {
        auto op = std::make_unique<InputOp>(val);
        return std::make_shared<Node>(Node(std::move(op), {}));
    }
    inline std::shared_ptr<Node> constant(double val) { return constant(Tensor::scalar(val)); }

    class Mul : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] * inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad * inputs[1], grad * inputs[0]};    
        }
    };

    inline std::shared_ptr<Node> mul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<Mul>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }

    class Div : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] / inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad / inputs[1], -1.0 * grad * inputs[0] / (inputs[1] * inputs[1])};    
        }
    };

    inline std::shared_ptr<Node> div(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<Div>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }

    class Add : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] + inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad, grad};    
        }
    };

    inline std::shared_ptr<Node> add(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<Add>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }

    class Sub : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] - inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad, grad * -1.0};    

        }
    };

    inline std::shared_ptr<Node> sub(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<Sub>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }

    class MatMul : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return matmul(inputs[0], inputs[1]);
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            if (inputs[0].shape().size() == 0 && inputs[1].shape().size() == 0) {
                return {grad * inputs[1], grad * inputs[0]};
            }

            Tensor grad_a, grad_b; 
            auto gshape = grad.shape();
            auto bshape = inputs[1].shape();

            if (gshape.size() == 1 && bshape.size() == 1) { // Compute the outer product for grad_a
                Tensor g = grad.reshape({gshape[0], 1}); // {n} -> {n,1}
                Tensor x = inputs[1].reshape({1, bshape[0]}); // {m} -> {1,m}
                grad_a = matmul2d(g, x);
                grad_b = matmul(inputs[0].transpose(), grad);
            }
            else {
                grad_a = matmul(grad, inputs[1].transpose());
                grad_b = matmul(inputs[0].transpose(), grad);
            }

            return {grad_a, grad_b};    
        }
    };

    inline std::shared_ptr<Node> matmul(std::shared_ptr<Node> a, std::shared_ptr<Node> b) {
        auto op = std::make_unique<MatMul>();
        return std::make_shared<Node>(Node(std::move(op), {a, b}));
    }

    class Sigmoid : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            Tensor copy = inputs[0];
            copy.map([](double v) {
                return 1.0 / (1.0 + std::exp(-v));
            });
            return copy;
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            auto val = forward(inputs);
            return {grad * val * (1.0 - val)};    
        }
    };

    inline std::shared_ptr<Node> sigmoid(std::shared_ptr<Node> a) {
        auto op = std::make_unique<Sigmoid>();
        return std::make_shared<Node>(Node(std::move(op), {a}));
    }

    class ReLU : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            Tensor copy = inputs[0];
            copy.map([](double v) {
                return std::max(v, 0.0);
            });
            return copy;
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            Tensor mask = inputs[0];
            mask.map([](double v) {
                return v > 0.0 ? 1.0 : 0.0;
            });
            return {grad * mask};    
        }
    };

    inline std::shared_ptr<Node> relu(std::shared_ptr<Node> a) {
        auto op = std::make_unique<ReLU>();
        return std::make_shared<Node>(Node(std::move(op), {a}));
    }

    class Tanh : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            Tensor copy = inputs[0];
            copy.map([](double v) {
                double ve = std::exp(2.0 * v);
                return (ve - 1.0) / (ve + 1.0);
            });
            return copy;
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            auto val = forward(inputs);
            return {grad * (Tensor::ones(val.shape()) - val * val)};    
        }
    };

    inline std::shared_ptr<Node> tanh(std::shared_ptr<Node> a) {
        auto op = std::make_unique<Tanh>();
        return std::make_shared<Node>(Node(std::move(op), {a}));
    }

    class HardTanh : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            Tensor copy = inputs[0];
            copy.map([](double v) {
                return std::max(std::min(v, 1.0), 1.0);
            });
            return copy;
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            Tensor mask = inputs[0];
            mask.map([](double v) {
                return v >= -1.0 && v <= 1.0 ? 1.0 : 0.0;
            });
            return {grad * mask};    
        }
    };

    inline std::shared_ptr<Node> hard_tanh(std::shared_ptr<Node> a) {
        auto op = std::make_unique<HardTanh>();
        return std::make_shared<Node>(Node(std::move(op), {a}));
    }
}
