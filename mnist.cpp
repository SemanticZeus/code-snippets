#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>
#include <iostream>
#include <cmath>
#include <stdexcept>

namespace ublas = boost::numeric::ublas;

// Activation functions
inline double sigmoid(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}
inline double sigmoid_derivative(double x) {
  double s = sigmoid(x);
  return s * (1.0 - s);
}

// Apply element-wise operation
template <typename Func>
ublas::matrix<double> apply_elementwise(const ublas::matrix<double>& mat, Func func) {
  ublas::matrix<double> result(mat.size1(), mat.size2());
  for (std::size_t i = 0; i < mat.size1(); ++i)
    for (std::size_t j = 0; j < mat.size2(); ++j)
      result(i, j) = func(mat(i, j));
  return result;
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

// Sum along rows or columns
ublas::vector<double> sum_matrix(const ublas::matrix<double>& mat, std::size_t axis) {
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

// Add bias vector to each column of the matrix
ublas::matrix<double> add_bias(const ublas::matrix<double>& mat, const ublas::vector<double>& bias) {
  ublas::matrix<double> result = mat;
  for (std::size_t j = 0; j < mat.size2(); ++j)
    for (std::size_t i = 0; i < mat.size1(); ++i)
      result(i, j) += bias(i);
  return result;
}

// Neural network class
class NeuralNetwork {
private:
  std::size_t input_size, hidden_size, output_size;
  ublas::matrix<double> W1, W2;  // Weights
  ublas::vector<double> b1, b2;  // Biases

public:
  NeuralNetwork(std::size_t input_size, std::size_t hidden_size, std::size_t output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size),
      W1(random_matrix(hidden_size, input_size)),
      W2(random_matrix(output_size, hidden_size)),
      b1(hidden_size, 0.0),
      b2(output_size, 0.0) {}

  // Forward propagation
  std::pair<ublas::matrix<double>, ublas::matrix<double>> forward(const ublas::matrix<double>& X) {
    // Z1 = W1 * X + b1
    ublas::matrix<double> Z1 = add_bias(ublas::prod(W1, X), b1);
    ublas::matrix<double> A1 = apply_elementwise(Z1, sigmoid);

    // Z2 = W2 * A1 + b2
    ublas::matrix<double> Z2 = add_bias(ublas::prod(W2, A1), b2);
    ublas::matrix<double> A2 = apply_elementwise(Z2, sigmoid);

    return {A1, A2};
  }

  // Train the network
  void train(const ublas::matrix<double>& X, const ublas::matrix<double>& y, std::size_t epochs, double lr) {
    std::size_t m = X.size2(); // Number of samples

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
      // Forward pass
      auto [A1, A2] = forward(X);

      // Compute loss (Mean Squared Error)
      double loss = 0.0;
      for (std::size_t i = 0; i < A2.size1(); ++i) {
        for (std::size_t j = 0; j < A2.size2(); ++j) {
          double diff = A2(i, j) - y(i, j);
          loss += diff * diff;
        }
      }
      loss /= 2.0 * A2.size2(); // Average loss per sample
      // Backpropagation
      ublas::matrix<double> dZ2 = (A2 - y) * 2.0 / m;
      ublas::matrix<double> dW2 = ublas::prod(dZ2, ublas::trans(A1));
      ublas::vector<double> db2 = sum_matrix(dZ2, 1);

      ublas::matrix<double> dA1 = ublas::prod(ublas::trans(W2), dZ2);
      ublas::matrix<double> dZ1 = dA1;
      for (std::size_t i = 0; i < dZ1.size1(); ++i)
        for (std::size_t j = 0; j < dZ1.size2(); ++j)
          dZ1(i, j) *= sigmoid_derivative(A1(i, j));

      ublas::matrix<double> dW1 = ublas::prod(dZ1, ublas::trans(X));
      ublas::vector<double> db1 = sum_matrix(dZ1, 1);

      // Gradient descent update
      W1 -= lr * dW1;
      W2 -= lr * dW2;
      b1 -= lr * db1;
      b2 -= lr * db2;

      // Print loss every 100 epochs
      if (epoch % 100 == 0) {
        std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
      }
    }
  }
};
#include <boost/filesystem.hpp>
#include <curl/curl.h>
#include <fstream>
#include <iostream>

// Function to download a file using libcurl
bool download_file(const std::string& url, const std::string& output_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "Failed to initialize CURL\n";
        return false;
    }

    FILE* fp = fopen(output_path.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open file for writing: " << output_path << "\n";
        curl_easy_cleanup(curl);
        return false;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    CURLcode res = curl_easy_perform(curl);
    fclose(fp);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cerr << "Download failed: " << curl_easy_strerror(res) << "\n";
        return false;
    }

    std::cout << "Downloaded: " << output_path << "\n";
    return true;
}

// Function to download and prepare MNIST data
void prepare_mnist() {
    namespace fs = boost::filesystem;

    // URLs for MNIST data
    const std::vector<std::pair<std::string, std::string>> mnist_files = {
        {"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"},
        {"http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"},
        {"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"},
        {"http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"}
    };

    // Directory to store downloaded files
    fs::path mnist_dir = "mnist_data";
    if (!fs::exists(mnist_dir)) {
        fs::create_directory(mnist_dir);
    }

    // Download files if not already present
    for (const auto& [url, filename] : mnist_files) {
        fs::path file_path = mnist_dir / filename;
        if (!fs::exists(file_path)) {
            std::cout << "File not found locally, downloading: " << filename << "\n";
            if (!download_file(url, file_path.string())) {
                std::cerr << "Failed to download " << filename << "\n";
                exit(EXIT_FAILURE);
            }
        } else {
            std::cout << "File already exists: " << filename << "\n";
        }
    }

    std::cout << "MNIST dataset is ready in the directory: " << mnist_dir.string() << "\n";
}

int main() {
    // Prepare MNIST dataset
    prepare_mnist();

    // Load the MNIST data from the downloaded files
    // (This part assumes you have a separate function to parse and load the MNIST files into matrices)
    std::size_t input_size = 784;    // 28x28 flattened
    std::size_t hidden_size = 128;  // Hidden layer size
    std::size_t output_size = 10;   // Digits 0-9

    std::size_t num_samples = 1000;  // Replace with the actual number of samples after parsing
    ublas::matrix<double> X(input_size, num_samples); // Training data
    ublas::matrix<double> y(output_size, num_samples); // Labels

    // Initialize neural network
    NeuralNetwork nn(input_size, hidden_size, output_size);

    // Train the network
    nn.train(X, y, 1000, 0.01); // Train for 1000 epochs with a learning rate of 0.01

    return 0;
}
