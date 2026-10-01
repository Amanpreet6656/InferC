#include <iostream>
#include <fstream>
#include "InferC.hpp"

using namespace inferc;

int main() {
    std::cout << "[INFO] Initializing InferC Engine...\n";

    // 1. Define Network Architecture
    // Example: Input(4) -> Dense(16) -> ReLU -> Dense(3) -> Softmax
    const size_t INPUT_DIM = 4;
    const size_t HIDDEN_DIM = 16;
    const size_t OUTPUT_DIM = 3;

    DenseLayer layer1(INPUT_DIM, HIDDEN_DIM);
    DenseLayer layer2(HIDDEN_DIM, OUTPUT_DIM);

    // 2. Load Pre-trained Weights from Python Export
    std::cout << "[INFO] Loading weights from model_weights.bin...\n";
    std::ifstream weight_file("model_weights.bin", std::ios::binary);
    
    if (!weight_file.is_open()) {
        std::cerr << "[ERROR] Could not open model_weights.bin! Run the Python export script first.\n";
        return 1;
    }

    try {
        layer1.load_weights(weight_file);
        layer2.load_weights(weight_file);
        std::cout << "[INFO] Weights loaded successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load weights: " << e.what() << "\n";
        return 1;
    }
    weight_file.close();

    // 3. Create Dummy Input (e.g., Batch Size = 2, Features = 4)
    Tensor input(2, INPUT_DIM);
    // Sample 1
    input(0, 0) = 1.0f; input(0, 1) = 0.5f; input(0, 2) = -1.2f; input(0, 3) = 0.1f;
    // Sample 2
    input(1, 0) = 0.0f; input(1, 1) = -0.5f; input(1, 2) = 2.0f; input(1, 3) = 1.1f;

    std::cout << "[INFO] Running Inference...\n";

    // 4. Forward Pass
    // X * W1 + B1
    Tensor hidden = layer1.forward(input);
    
    // ReLU Activation
    activation::relu(hidden);
    
    // H * W2 + B2
    Tensor output = layer2.forward(hidden);
    
    // Softmax Probabilities
    activation::softmax(output);

    // 5. Output Results
    std::cout << "\n[RESULT] Inference Probabilities:\n";
    for (size_t i = 0; i < output.rows(); ++i) {
        std::cout << "Sample " << i << ": [ ";
        for (size_t j = 0; j < output.cols(); ++j) {
            std::cout << output(i, j) << " ";
        }
        std::cout << "]\n";
    }

    return 0;
}
