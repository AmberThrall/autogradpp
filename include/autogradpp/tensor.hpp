#pragma once

#include <iostream>
#include <random>
#include <stdexcept>
#include <functional>
#include <vector>

namespace autogradpp {
    using Index = std::vector<size_t>;

    class Tensor {
    public:
        using iterator = typename std::vector<double>::iterator;

        Tensor() : _shape({}), _data(compute_size(_shape)) {
            compute_strides();
        }

        Tensor(Index shape) : _shape(std::move(shape)), _data(compute_size(_shape)) {
            compute_strides();
        }

        /// Returns a scalar tensor
        static Tensor scalar(double val) { 
            Tensor t;
            t[0] = val;
            return t;
        }

        /// Returns a tensor filled with zeros
        static Tensor zeros(Index shape) { return Tensor(shape); }

        /// Returns a 1-tensor given by arguments
        static Tensor vector(std::vector<double> entries) {
            Tensor t({entries.size()});
            for (size_t i = 0; i < entries.size(); ++i) {
                t[i] = entries[i];
            }
            return t;
        }

        /// Returns a tensor filled with ones 
        static Tensor ones(Index shape) {
            Tensor t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t[k] = 1;
            }
            return t;
        }

        /// Creates a 2-D tensor with ones on the diagonal and zeros elsewhere
        static Tensor eye(size_t n) {
            Tensor t({n, n});
            for (size_t k = 0; k < n; ++k) {
                t(k,k) = 1;
            }
            return t;
        }

        /// Returns a tensor filled with uniform random values on [low,high) 
        static Tensor rand(Index shape, double low = 0.0, double high = 1.0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> distrib(low, high);

            Tensor t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t[k] = distrib(gen);
            }
            return t;
        }

        /// Returns a tensor filled with uniform random integers on [low, high]
        static Tensor randint(int low, int high, Index shape) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> distrib(low, high);

            Tensor t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t[k] = distrib(gen);
            }
            return t;
        }

        /// Returns a tensor filled with random values from the standard normal distribution
        static Tensor randn(Index shape, double mean = 0.0, double stddev = 1.0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> distrib(mean, stddev);

            Tensor t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t[k] = distrib(gen);
            }
            return t;
        }

        template<typename... Idx>
        double& operator()(Idx... idx) {
            return _data[offset({static_cast<size_t>(idx)...})];
        }

        template<typename... Idx>
        const double& operator()(Idx... idx) const {
            return _data[offset({static_cast<size_t>(idx)...})];
        }

        double& operator[](size_t idx) { return _data[idx]; }

        const double& operator[](size_t idx) const { return _data[idx]; }

        double& operator()(Index idx) {
            return _data[offset(idx)];
        }

        const double& operator()(Index idx) const {
            return _data[offset(idx)];
        }

        void map(std::function<double(double)> fn) {
            for (size_t k = 0; k < _data.size(); ++k) {
                _data[k] = fn(_data[k]); 
            }
        }

        const size_t rank() const { return _shape.size(); }
        const Index& shape() const { return _shape; }
        size_t size() const { return _data.size(); }
        const Index& strides() const { return _strides; }

        Tensor transpose() const {
            if (_shape.size() < 2) { return Tensor(*this); }
            if (_shape.size() > 2) {
                throw std::runtime_error("transpose: higher-dimensional transpose not supported.");
            }

            Tensor copy({_shape[1], _shape[0]});
            for (size_t r = 0; r < _shape[1]; ++r) {
                for (size_t c = 0; c < _shape[0]; ++c) {
                    double v = _data[c * _strides[0] + r * _strides[1]];
                    copy.data()[r * copy.strides()[0] + c * copy.strides()[1]] = v;
                }
            }

            return copy;
        }

        Tensor reshape(Index new_shape) const {
            size_t new_size = compute_size(new_shape); 
            if (new_size != _data.size()) { 
                throw std::invalid_argument("reshape: total size mismatch.");
            }

            Tensor res;
            res._shape = std::move(new_shape);
            res._data = _data;
            res.compute_strides();
            return res;
        }

        double* data() { return _data.data(); }
        const double* data() const { return _data.data(); }

        iterator begin() { return _data.begin(); }
        iterator end() { return _data.end(); }

        // Operator overloads
        bool operator==(const Tensor& rhs) const {
            if (shape() != rhs.shape()) { return false; }
            return _data == rhs._data;
        }
        bool operator!=(const Tensor& rhs) const { return !(*this == rhs); }

        operator double() const {
            if (rank() != 0) { 
                throw std::invalid_argument("can only convert 0-tensor to double.");
            }
            return _data[0];
        }
       
        Tensor& operator+=(const Tensor& rhs) {
            if (shape() != rhs.shape()) {
                throw std::invalid_argument("cannot add tensors of different shape."); 
            }
            for (size_t k = 0; k < size(); ++k) {
                _data[k] += rhs[k];
            }
            return *this;
        }

        Tensor& operator-=(const Tensor& rhs) {
            if (shape() != rhs.shape()) {
                throw std::invalid_argument("cannot subtract tensors of different shape."); 
            }
            for (size_t k = 0; k < size(); ++k) {
                _data[k] -= rhs[k];
            }
            return *this;
        }

        Tensor& operator*=(const Tensor& rhs) {
            if (shape() != rhs.shape() && rhs.shape().size() != 0) {
                throw std::invalid_argument("cannot multiply tensors of different shape."); 
            }
            for (size_t k = 0; k < size(); ++k) {
                if (rhs.shape().size() == 0) {
                    _data[k] *= rhs[0];
                }
                else {
                    _data[k] *= rhs[k];
                }
            }
            return *this;
        }

        Tensor& operator/=(const Tensor& rhs) {
            if (shape() != rhs.shape() && rhs.shape().size() != 1) {
                throw std::invalid_argument("cannot divide tensors of different shape."); 
            }
            for (size_t k = 0; k < size(); ++k) {
                if (rhs.shape().size() == 1) {
                    _data[k] /= rhs[0];
                }
                else {
                    _data[k] /= rhs[k];
                }
            }
            return *this;
        }

        Tensor& operator*=(double rhs) {
            for (size_t k = 0; k < size(); ++k) {
                _data[k] *= rhs;
            }
            return *this;
        }

        Tensor& operator/=(double rhs) {
            for (size_t k = 0; k < size(); ++k) {
                _data[k] /= rhs;
            }
            return *this;
        }
    private:
        Index _shape;
        Index _strides;
        std::vector<double> _data;

        static size_t compute_size(Index& shape) {
            size_t s = 1;
            for (auto d : shape) { s *= d; }
            return s;
        }

        void compute_strides() {
            _strides.resize(_shape.size());
            size_t stride = 1;
            for (int i = static_cast<int>(_shape.size()) - 1; i >= 0; --i) {
                _strides[i] = stride;
                stride *= _shape[i];
            }
        }

        size_t offset(Index idx) const {
            size_t off = 0;
            size_t i = 0;
            for (auto v : idx) { off += v * _strides[i++]; }
            return off;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        const Index& shape = t.shape();
        size_t rank = shape.size();
        size_t total = t.size();

        if (total == 0) { os << "[]"; return os; }
        Index idx(rank, 0);
        for (size_t i = 0; i < rank; ++i) { os << "["; }

        for (size_t k = 0; k < total; ++k) {
            os << t(idx);

            if (k+1 == total) { break; }

            size_t closed = 0; // how many dimensions rolled over
            for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
                idx[i]++;
                if (idx[i] < shape[i]) break;
                idx[i] = 0;
                closed++;
            }

            for (size_t i = 0; i < closed; ++i) { os << "]"; }
            os << ",";
            if (closed > 0) {
                for (size_t i = 0; i < closed; ++i) { os << "["; }
            }
        }

        for (size_t i = 0; i < rank; ++i) { os << "]"; }

        return os;
    }

    inline Tensor operator+(Tensor lhs, const Tensor& rhs) { lhs += rhs; return lhs; }
    inline Tensor operator-(Tensor lhs, const Tensor& rhs) { lhs -= rhs; return lhs; }
    inline Tensor operator*(Tensor lhs, Tensor rhs) { 
        if (lhs.rank() == 0) {
            rhs *= lhs; return rhs; 
        }
        lhs *= rhs; return lhs; 
    }
    inline Tensor operator/(Tensor lhs, const Tensor& rhs) { lhs /= rhs; return lhs; }
    inline Tensor operator*(Tensor lhs, double rhs) { lhs *= rhs; return lhs; }
    inline Tensor operator*(double rhs, Tensor lhs) { lhs *= rhs; return lhs; }
    inline Tensor operator/(Tensor lhs, double rhs) { lhs /= rhs; return lhs; }

    inline Tensor matmul2d(const Tensor& lhs, const Tensor& rhs) {
        const Index& ashape = lhs.shape();
        const Index& bshape = rhs.shape();
        if (ashape.size() != 2 || bshape.size() != 2) {
            throw std::invalid_argument("matmul2d: tensors must be 2-D.");
        }

        if (ashape[1] != bshape[0]) {
            throw std::invalid_argument("matmul2d: inner dimensions don't match.");
        }

        Tensor out({ashape[0], bshape[1]});
        for (size_t i = 0; i < ashape[0]; ++i) {
            for (size_t j = 0; j < bshape[1]; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < ashape[1]; ++k) {
                    double a = lhs.data()[i * lhs.strides()[0] + k * lhs.strides()[1]];
                    double b = rhs.data()[k * rhs.strides()[0] + j * rhs.strides()[1]];
                    sum += a * b;
                    //sum += lhs(i, k) * rhs(k, j);
                }

                out(i, j) = sum;
            }
        }

        return out;
    }
   
    inline Tensor matmul(Tensor lhs, Tensor rhs) {
        Index ashape = lhs.shape();
        Index bshape = rhs.shape();
        if (ashape.size() > 2 || bshape.size() > 2) {
            throw std::invalid_argument("matmul: higher dimensional matrix multiplication is currently not supported.");
        }
        /*if (ashape.size() == 1) { std::cout << "A: " << ashape[0] << std::endl; } 
        else if (ashape.size() == 2) { std::cout << "A: " << ashape[0] << "x" << ashape[1] << std::endl; }
        if (bshape.size() == 1) { std::cout << "B: " << bshape[0] << std::endl; } 
        else if (bshape.size() == 2) { std::cout << "B: " << bshape[0] << "x" << bshape[1] << std::endl; }*/

        if (ashape.size() == 0) { return rhs * lhs; }
        if (bshape.size() == 0) { return lhs * rhs; }
        if (ashape.size() == 1) { lhs = lhs.reshape({1, ashape[0]}); }
        if (bshape.size() == 1) { rhs = rhs.reshape({bshape[0], 1}); }

        Tensor res = matmul2d(lhs, rhs);
        if (ashape.size() == 1 && bshape.size() == 1) { return res.reshape({}); }
        else if (ashape.size() == 1) { 
            Index s = res.shape();
            s.erase(s.end() - 2); 
            return res.reshape(s);
        }
        else if (bshape.size() == 1) { 
            Index s = res.shape();
            s.erase(s.end() - 1); 
            return res.reshape(s);
        }

        return res;
    }
}
