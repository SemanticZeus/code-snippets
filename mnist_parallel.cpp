#include <boost/thread.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <random>
#include <cmath>
#include <iostream>

namespace ublas = boost::numeric::ublas;

// Activation functions
inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline double sigmoid_derivative(double x) { double s = sigmoid(x); return s * (1.0 - s); }

// Apply element-wise operation
template <typename Func>
ublas::matrix<double> apply_elementwise(const ublas::matrix<double>& mat, Func func) {
    ublas::matrix<double> result(mat.size1(), mat.size2());
    for (std::size_t i = 0; i < mat.size1(); ++i)
        for (std::size_t j = 0; j < mat.size2(); ++j)
            result(i, j) = func(mat(i, j));
    return result;
}

// Sum along rows or columns
ublas::vector<double> sum(const ublas::matrix<double>& mat, std::size_t axis) {
    if (axis == 0) { // Sum along columns
        ublas::vector<double> result(mat.size2(), 0.0);
        for (std::size_t i = 0; i < mat.size1(); ++i)
            for (std::size_t j = 0; j < mat.size2(); ++j)
                result(j) += mat(i, j);
        return result;
    } else if (axis == 1) { // Sum along rows
        ublas::vector<double> result(mat.size1(), 0.0);
        for (std::size_t i = 0; i < mat.size1(); ++i)
            for (std::size_t j = 0; j < mat.size2(); ++j)
                result(i) += mat(i, j);
        return result;
    }
    throw std::invalid_argument("Invalid axis, must be 0 or 1.");
}

// Randomly initialize weights
ublas::matrix<double> random_matrix(std::size_t rows, std::size_t cols, double min = -0.5, double max = 0.5) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(min, max);

    ublas::matrix<double> mat(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            mat(i, j) = dist(gen);
    return mat;
}

// Neural network class
class NeuralNetwork {
private:
    std::size_t input_size, hidden_size, output_size;
    ublas::matrix<double> W1, W2;
    ublas::vector<double> b1, b2; // Biases as vectors

public:
    NeuralNetwork(std::size_t input_size, std::size_t hidden_size, std::size_t output_size)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
          W1(random_matrix(hidden_size, input_size)), 
          W2(random_matrix(output_size, hidden_size)),
          b1(hidden_size, 0.0),
          b2(output_size, 0.0) {}

    // Forward propagation
    std::pair<ublas::matrix<double>, ublas::matrix<double>> forward(const ublas::matrix<double>& X) {
        auto Z1 = ublas::prod(W1, X);

    for (std::size_t i = 0; i < Z1.size1(); ++i)
      for (std::size_t j = 0; j < Z1.size2(); ++j)
        Z1(i, j) += b1(i);


        auto A1 = apply_elementwise(Z1, sigmoid);

        auto Z2 = ublas::prod(W2, A1);


    for (std::size_t i = 0; i < Z2.size1(); ++i)
      for (std::size_t j = 0; j < Z2.size2(); ++j)
        Z2(i, j) = Z2(i, j) + b2(i);

        auto A2 = apply_elementwise(Z2, sigmoid);
        return {A1, A2};
    }

    // Train the network
    void train(const ublas::matrix<double>& X, const ublas::matrix<double>& y, std::size_t epochs, double lr) {
        std::size_t num_threads = boost::thread::hardware_concurrency();
        std::cout << "Using " << num_threads << " threads for training.\n";

        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            auto [A1, A2] = forward(X);

            // Backpropagation
            ublas::matrix<double> dZ2 = A2 - y;
            ublas::matrix<double> dW2 = ublas::zero_matrix<double>(W2.size1(), W2.size2());
            ublas::matrix<double> dW1 = ublas::zero_matrix<double>(W1.size1(), W1.size2());
            ublas::vector<double> db2(W2.size1(), 0.0);
            ublas::vector<double> db1(W1.size1(), 0.0);

            boost::thread_group threads;

            // Parallelize computations for gradients
            for (std::size_t t = 0; t < num_threads; ++t) {
                threads.create_thread([&, t]() {
                    for (std::size_t i = t; i < X.size2(); i += num_threads) {
                        auto col = ublas::column(X, i);
                        auto label = ublas::column(y, i);

                        // Compute partial gradients
                        auto z1 = ublas::prod(W1, col) + b1;
                        auto a1 = apply_elementwise(z1, sigmoid);

                        auto z2 = ublas::prod(W2, a1) + b2;
                        auto a2 = apply_elementwise(z2, sigmoid);

                        auto dz2 = a2 - label;
                        auto dw2 = ublas::outer_prod(dz2, a1);

                        auto dz1 = ublas::prod(ublas::trans(W2), dz2) * apply_elementwise(z1, sigmoid_derivative);
                        auto dw1 = ublas::outer_prod(dz1, col);

                        // Aggregate updates
                        for (std::size_t j = 0; j < W2.size1(); ++j) {
                            db2(j) += dz2(j);
                            for (std::size_t k = 0; k < W2.size2(); ++k)
                                dW2(j, k) += dw2(j, k);
                        }
                        for (std::size_t j = 0; j < W1.size1(); ++j) {
                            db1(j) += dz1(j);
                            for (std::size_t k = 0; k < W1.size2(); ++k)
                                dW1(j, k) += dw1(j, k);
                        }
                    }
                });
            }

            threads.join_all();

            // Update weights and biases
            W1 -= lr * dW1;
            W2 -= lr * dW2;
            for (std::size_t i = 0; i < b1.size(); ++i) b1(i) -= lr * db1(i);
            for (std::size_t i = 0; i < b2.size(); ++i) b2(i) -= lr * db2(i);

            // Print loss every 100 epochs
            if (epoch % 100 == 0) {
                double loss = 0.0;
                for (std::size_t i = 0; i < A2.size1(); ++i) {
                    for (std::size_t j = 0; j < A2.size2(); ++j) {
                        double diff = A2(i, j) - y(i, j);
                        loss += diff * diff;
                    }
                }
                loss /= 2.0 * A2.size2(); // Average loss
                std::cout << "Epoch " << epoch << ", Loss: " << loss << "\n";
            }
        }
    }
};

int main() {
    // Dummy MNIST-like data
    std::size_t input_size = 784;   // 28x28 flattened
    std::size_t hidden_size = 128; // Hidden layer size
    std::size_t output_size = 10;  // Digits 0-9

    std::size_t num_samples = 1000;
    ublas::matrix<double> X(input_size, num_samples, 0.1); // Dummy inputs
    ublas::matrix<double> y(output_size, num_samples, 0.0); // Dummy labels

    // Initialize neural network
    NeuralNetwork nn(input_size, hidden_size, output_size);

    // Train the network
    nn.train(X, y, 1000, 0.01); // 1000 epochs, 0.01 learning rate

    return 0;
}

