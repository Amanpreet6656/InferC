#ifndef INFERC_HPP
#define INFERC_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <memory>

namespace inferc {

/**
 * @class Tensor
 * @brief A lightweight, 2D continuous memory block for matrix operations.
 * 
 * Uses std::vector under the hood to ensure RAII memory management.
 * Elements are stored in row-major order to optimize for spatial locality
 * during CPU cache line fetching.
 */
class Tensor {
private:
    size_t m_rows;
    size_t m_cols;
    std::vector<float> m_data;

public:
    Tensor() : m_rows(0), m_cols(0) {}
    
    Tensor(size_t rows, size_t cols, float init_val = 0.0f) 
        : m_rows(rows), m_cols(cols), m_data(rows * cols, init_val) {}

    // Expose dimensions
    inline size_t rows() const { return m_rows; }
    inline size_t cols() const { return m_cols; }
    inline size_t size() const { return m_data.size(); }
    
    // Raw data access
    inline float* data() { return m_data.data(); }
    inline const float* data() const { return m_data.data(); }

    // 2D access via row-major index calculation
    inline float& operator()(size_t r, size_t c) {
        return m_data[r * m_cols + c];
    }
    
    inline const float& operator()(size_t r, size_t c) const {
        return m_data[r * m_cols + c];
    }

    /**
     * @brief Cache-friendly Matrix Multiplication (C = A * B)
     * 
     * Time Complexity: O(I * J * K)
     * Space Complexity: O(I * J) for the resulting tensor.
     * 
     * Optimization: We use the i-k-j loop ordering instead of the naive i-j-k.
     * By placing 'j' in the innermost loop, we traverse both the current row of A 
     * and the current row of B sequentially in memory. This maximizes spatial 
     * locality and drastically reduces L1/L2 cache misses compared to column-wise traversal.
     */
    Tensor matmul(const Tensor& other) const {
        if (this->m_cols != other.m_rows) {
            throw std::invalid_argument("Matrix dimension mismatch for multiplication.");
        }

        Tensor result(this->m_rows, other.m_cols, 0.0f);

        // i-k-j loop order for cache locality
        for (size_t i = 0; i < this->m_rows; ++i) {
            for (size_t k = 0; k < this->m_cols; ++k) {
                float a_ik = (*this)(i, k);
                for (size_t j = 0; j < other.m_cols; ++j) {
                    result(i, j) += a_ik * other(k, j);
                }
            }
        }
        return result;
    }

    /**
     * @brief Row-wise addition (Broadcasting bias to batch size)
     */
    void add_bias(const Tensor& bias) {
        if (bias.rows() != 1 || bias.cols() != this->m_cols) {
            throw std::invalid_argument("Bias dimension mismatch.");
        }
        for (size_t i = 0; i < this->m_rows; ++i) {
            for (size_t j = 0; j < this->m_cols; ++j) {
                (*this)(i, j) += bias(0, j);
            }
        }
    }
};

/**
 * @class DenseLayer
 * @brief Fully Connected Layer computing Y = XW + B
 */
class DenseLayer {
private:
    Tensor weights;
    Tensor biases;

public:
    DenseLayer(size_t input_dim, size_t output_dim) 
        : weights(input_dim, output_dim), biases(1, output_dim) {}

    /**
     * @brief Deserializes weights and biases from a binary file stream.
     * Assumes contiguous float32 binary format written by the Python exporter.
     */
    void load_weights(std::ifstream& is) {
        int32_t in_dim = 0, out_dim = 0;
        is.read(reinterpret_cast<char*>(&in_dim), sizeof(int32_t));
        is.read(reinterpret_cast<char*>(&out_dim), sizeof(int32_t));

        if (in_dim != static_cast<int32_t>(weights.rows()) || out_dim != static_cast<int32_t>(weights.cols())) {
            throw std::runtime_error("Weight dimension mismatch during loading.");
        }

        // Read Weights
        is.read(reinterpret_cast<char*>(weights.data()), weights.size() * sizeof(float));
        // Read Biases
        is.read(reinterpret_cast<char*>(biases.data()), biases.size() * sizeof(float));
    }

    /**
     * @brief Forward pass: computes X * W + B
     */
    Tensor forward(const Tensor& input) const {
        Tensor output = input.matmul(weights);
        output.add_bias(biases);
        return output;
    }
};

namespace activation {

    /**
     * @brief Rectified Linear Unit (ReLU)
     * Mathematical definition: f(x) = max(0, x)
     * Operates in-place on the input tensor.
     */
    inline void relu(Tensor& t) {
        for (size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = std::max(0.0f, t.data()[i]);
        }
    }

    /**
     * @brief Softmax Activation
     * Mathematical definition: f(x_i) = exp(x_i) / sum(exp(x_j))
     * Includes numerical stability improvement by subtracting the max value 
     * in each row before exponentiation to prevent float overflow.
     * Operates in-place on the input tensor.
     */
    inline void softmax(Tensor& t) {
        for (size_t i = 0; i < t.rows(); ++i) {
            // Find max for numerical stability
            float max_val = -INFINITY;
            for (size_t j = 0; j < t.cols(); ++j) {
                if (t(i, j) > max_val) max_val = t(i, j);
            }

            // Exponentiate and sum
            float sum_exp = 0.0f;
            for (size_t j = 0; j < t.cols(); ++j) {
                t(i, j) = std::exp(t(i, j) - max_val);
                sum_exp += t(i, j);
            }

            // Normalize
            for (size_t j = 0; j < t.cols(); ++j) {
                t(i, j) /= sum_exp;
            }
        }
    }

} // namespace activation

} // namespace inferc

#endif // INFERC_HPP
