#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include <cmath>
#include <memory>

namespace autogradpp {
    /// Base operand for input values
    class InputOp : public Operand {
    public:
        InputOp(Tensor val) : _val(val) {}

        Tensor forward(std::vector<Tensor>& inputs) {
            return _val;
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {};    
        }
        
        std::string name() const { return "Input"; }
    private:
        Tensor _val;
    };

    /// Operand for x + y
    class Add : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] + inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad, grad};    
        }
        
        std::string name() const { return "Add"; }
    };

    /// Operand for x - y 
    class Sub : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] - inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad, grad * -1.0};    

        }

        std::string name() const { return "Sub"; }
    };

    /// Operand for x * y
    class Mul : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] * inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad * inputs[1], grad * inputs[0]};    
        }

        std::string name() const { return "Mul"; }
    };

    /// Operand for x / y
    class Div : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            return inputs[0] / inputs[1];
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            return {grad / inputs[1], -1.0 * grad * inputs[0] / (inputs[1] * inputs[1])};    
        }

        std::string name() const { return "Div"; }
    };

    /// Thin wrapper for creating computational graphs
    class Variable {
    public:
        Variable(double val = 0.0, bool requires_grad = true) : Variable(Tensor::scalar(val), requires_grad) {}
        Variable(Tensor val, bool requires_grad = true) {
            auto op = std::make_unique<InputOp>(val);
            _node = std::make_shared<Node>(Node(std::move(op), {}, requires_grad));
        }
        Variable(std::shared_ptr<Node> node) : _node(node) {}

        Index shape() const { return _node->value.shape(); }
        std::shared_ptr<Node> node() { return _node; }
        Tensor& value() { return _node->value; }
        const Tensor& value() const { return _node->value; }
        Tensor& grad() { return _node->grad; }
        const Tensor& grad() const { return _node->grad; }

        void backward() { _node->backward(); }

        // Operator overloads
        Variable& operator+=(const Variable& rhs) {
            if (shape() != rhs.shape()) {
                throw std::invalid_argument("cannot add tensors of different shape."); 
            }

            auto op = std::make_unique<Add>();
            auto new_node = std::make_shared<Node>(Node(std::move(op), {_node, rhs._node}));
            _node = new_node;
            return *this;
        }

        Variable& operator-=(const Variable& rhs) {
            if (shape() != rhs.shape()) {
                throw std::invalid_argument("cannot subtract tensors of different shape."); 
            }

            auto op = std::make_unique<Sub>();
            auto new_node = std::make_shared<Node>(Node(std::move(op), {_node, rhs._node}));
            _node = new_node;
            return *this;
        }

        Variable& operator*=(const Variable& rhs) {
            if (shape() != rhs.shape() && rhs.shape().size() != 0) {
                throw std::invalid_argument("cannot multiply tensors of different shape."); 
            }

            auto op = std::make_unique<Mul>();
            auto new_node = std::make_shared<Node>(Node(std::move(op), {_node, rhs._node}));
            _node = new_node;
            return *this;
        }
        Variable& operator*=(Tensor rhs) { *this *= Variable(rhs, false); return *this; }
        Variable& operator*=(double rhs) { *this *= Variable(rhs, false); return *this; }

        Variable& operator/=(const Variable& rhs) {
            if (shape() != rhs.shape() && rhs.shape().size() != 0) {
                throw std::invalid_argument("cannot divide tensors of different shape."); 
            }

            auto op = std::make_unique<Div>();
            auto new_node = std::make_shared<Node>(Node(std::move(op), {_node, rhs._node}));
            _node = new_node;
            return *this;
        }
        
        Variable& operator/=(Tensor rhs) { *this /= Variable(rhs, false); return *this; }
        Variable& operator/=(double rhs) { *this /= Variable(rhs, false); return *this; }
    private:
        std::shared_ptr<Node> _node;
    };

    /// Special case of variable for constants
    class Constant : public Variable {
        public:
            Constant(double val) : Constant(Tensor::scalar(val)) {}
            Constant(Tensor val) : Variable(val, false) {}
    };

    inline Variable operator+(const Variable& a, const Variable& b) { Variable copy = a; copy += b; return copy; }
    inline Variable operator-(const Variable& a, const Variable& b) { Variable copy = a; copy -= b; return copy; }
    inline Variable operator*(const Variable& a, const Variable& b) { Variable copy = a; copy *= b; return copy; }
    inline Variable operator*(const Variable& a, const Tensor& b) { Variable copy = a; copy *= Constant(b); return copy; }
    inline Variable operator*(const Variable& a, const double& b) { Variable copy = a; copy *= Constant(b); return copy; }
    inline Variable operator*(const Tensor& a, const Variable& b) { Variable copy = Constant(a); copy *= b; return copy; }
    inline Variable operator*(const double& a, const Variable& b) { Variable copy = Constant(a); copy *= b; return copy; }
    inline Variable operator/(const Variable& a, const Variable& b) { Variable copy = a; copy /= b; return copy; }
    inline Variable operator/(const Variable& a, const Tensor& b) { Variable copy = a; copy /= Constant(b); return copy; }
    inline Variable operator/(const Variable& a, const double& b) { Variable copy = a; copy /= Constant(b); return copy; }
    inline Variable operator/(const Tensor& a, const Variable& b) { Variable copy = Constant(a); copy /= b; return copy; }
    inline Variable operator/(const double& a, const Variable& b) { Variable copy = Constant(a); copy /= b; return copy; }

    /// Operand for matrix multiplication
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

        std::string name() const { return "MatMul"; }
    };

    inline Variable matmul(Variable a, Variable b) {
        Index ashape = a.shape();
        Index bshape = b.shape();
        if (ashape.size() > 2 || bshape.size() > 2) {
            throw std::invalid_argument("matmul: higher dimensional matrix multiplication is currently not supported.");
        }
        if (ashape.size() == 2 || bshape.size() == 2) {
            if (ashape.size() == 2 && ashape[1] != bshape[0]) {
                throw std::invalid_argument("matmul: inner dimensions don't match.");
            }
            if (ashape.size() == 1 && ashape[0] != bshape[0]) {
                throw std::invalid_argument("matmul: inner dimensions don't match.");
            }
        }

        auto op = std::make_unique<MatMul>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {a.node(), b.node()}));
        return Variable(new_node);
    }

    /// Operand for sigmoid activation function
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
            return {grad * val * (Tensor::ones(val.shape()) - val)};    
        }

        std::string name() const { return "Sigmoid"; }
    };

    inline Variable sigmoid(Variable a) {
        auto op = std::make_unique<Sigmoid>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {a.node()}));
        return Variable(new_node);  
    }

    /// Operand for ReLU activation function
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
            auto res = grad * mask;
            return {res};    
        }

        std::string name() const { return "ReLU"; }
    };

    inline Variable relu(Variable a) {
        auto op = std::make_unique<ReLU>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {a.node()}));
        return Variable(new_node);  
    }
    
    /// Operand for hyperbolic tangent activation function
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

        std::string name() const { return "Tanh"; }
    };

    inline Variable tanh(Variable a) {
        auto op = std::make_unique<Tanh>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {a.node()}));
        return Variable(new_node);  
    }
   
    /// Operand for hard hyperbolic tangent activation function
    class HardTanh : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            Tensor copy = inputs[0];
            copy.map([](double v) {
                return std::max(std::min(v, 1.0), -1.0);
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

        std::string name() const { return "HardTanh"; }
    };

    inline Variable hard_tanh(Variable a) {
        auto op = std::make_unique<HardTanh>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {a.node()}));
        return Variable(new_node);  
    }
  
    /// Operand for softmax function
    class Softmax : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) {
            if (inputs[0].rank() == 0) { return Tensor::scalar(1); }

            double max_val = *std::max_element(inputs[0].data(), inputs[0].data() + inputs[0].size());
            double den = 0;
            for (size_t i = 0; i < inputs[0].size(); ++i) {
                den += std::exp(inputs[0][i] - max_val);
            }
            Tensor copy = inputs[0];
            copy.map([&](double v) { return std::exp(v - max_val) / den; });
            return copy;
        }

        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) {
            auto a = forward(inputs);
            double dot = 0.0;
            for (size_t i = 0; i < a.size(); ++i) {
                dot += grad[i] * a[i];
            }
            Tensor result = a;
            for (size_t i = 0; i < result.size(); ++i) {
                result[i] = a[i] * (grad[i] - dot);
            }
            return {result};
        }

        std::string name() const { return "Softmax"; }
    };

    inline Variable softmax(Variable a) {
        auto op = std::make_unique<Softmax>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {a.node()}));
        return Variable(new_node);  
    }
 
    /// Operand for identity function
    class Linear : public Operand {
    public:
        Tensor forward(std::vector<Tensor>& inputs) { return inputs[0]; }
        std::vector<Tensor> backward(std::vector<Tensor>& inputs, const Tensor& grad) { return { grad.reshape(inputs[0].shape()) }; }
        std::string name() const { return "Linear"; }
    };

    inline Variable linear(Variable a) {
        auto op = std::make_unique<Linear>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {a.node()}));
        return Variable(new_node);  
    }

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

        std::string name() const { return "CrossEntropyLoss"; }
    };

    inline Variable cross_entropy(Variable y_hat, Variable y_true) {
        auto op = std::make_unique<CrossEntropyLoss>();
        auto new_node = std::make_shared<Node>(Node(std::move(op), {y_hat.node(), y_true.node()}));
        return Variable(new_node);
    }
}
