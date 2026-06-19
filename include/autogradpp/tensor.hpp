#pragma once

#include <iostream>
#include <random>
#include <vector>

namespace autogradpp {
    using Index = std::vector<size_t>;

    template <typename T>
    class Tensor {
    public:
        using iterator = typename std::vector<T>::iterator;

        Tensor(Index shape) : _shape(std::move(shape)), _data(compute_size(_shape)) {
            compute_strides();
        }

        /// Returns a tensor filled with zeros
        static Tensor<T> zeros(Index shape) { return Tensor<T>(shape); }

        /// Returns a tensor filled with ones 
        static Tensor<T> ones(Index shape) {
            Tensor<T> t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t.data()[k] = 1;
            }
            return t;
        }

        /// Creates a 2-D tensor with ones on the diagonal and zeros elsewhere
        static Tensor<T> eye(size_t n) {
            Tensor<T> t({n, n});
            for (size_t k = 0; k < n; ++k) {
                t(k,k) = 1;
            }
            return t;
        }

        /// Returns a tensor filled with uniform random values on [low,high) 
        static Tensor<T> rand(int low, int high, Index shape) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> distrib(low, high);

            Tensor<T> t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t.data()[k] = distrib(gen);
            }
            return t;
        }

        /// Returns a tensor filled with uniform random integers on [low, high]
        static Tensor<T> randint(int low, int high, Index shape) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> distrib(low, high);

            Tensor<T> t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t.data()[k] = distrib(gen);
            }
            return t;
        }

        /// Returns a tensor filled with random values from the standard normal distribution
        static Tensor<T> randn(Index shape) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> distrib(0, 1);

            Tensor<T> t(shape);
            for (size_t k = 0; k < t.size(); ++k) {
                t.data()[k] = distrib(gen);
            }
            return t;
        }

        template<typename... Idx>
        T& operator()(Idx... idx) {
            return _data[offset({static_cast<size_t>(idx)...})];
        }

        template<typename... Idx>
        const T& operator()(Idx... idx) const {
            return _data[offset({static_cast<size_t>(idx)...})];
        }

        T& operator()(Index idx) {
            return _data[offset(idx)];
        }

        const T& operator()(Index idx) const {
            return _data[offset(idx)];
        }

        const size_t rank() const { return _shape.size() - 1; }
        const Index& shape() const { return _shape; }
        size_t size() const { return _data.size(); }

        T* data() { return _data.data(); }
        const T* data() const { return _data.data(); }

        iterator begin() { return _data.begin(); }
        iterator end() { return _data.end(); }
    private:
        Index _shape;
        Index _strides;
        std::vector<T> _data;

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

    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
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

    using Tensorf = Tensor<float>;
    using Tensord = Tensor<double>;
}
