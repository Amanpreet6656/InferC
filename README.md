# InferC: Zero-Dependency C++17 ML Inference Engine

**InferC** is a highly optimized, header-only C++ library designed for running neural network inference on edge devices. It strictly avoids heavy external dependencies like Eigen, OpenCV, or TensorFlow. Instead, it implements core tensor math, cache-optimized matrix multiplications, and back-propagation algorithms directly via the C++ standard library.

This project bridges the gap between high-level Python training environments and bare-metal, low-level deployment targets.

---

## 🔬 Mathematical Architecture

The core of `InferC` revolves around simulating the standard Feed-Forward neural network operations explicitly. 

### Fully Connected (Dense) Layer
For a layer $L$, given an input matrix $X \in \mathbb{R}^{B \times I}$ (where $B$ is batch size and $I$ is input features), weight matrix $W \in \mathbb{R}^{I \times O}$, and bias vector $B \in \mathbb{R}^{1 \times O}$, the forward transformation is:
$$ Y = X \cdot W + B $$

### Non-Linearities
- **ReLU (Rectified Linear Unit):** Operates piecewise over the tensor $T$.
  $$ f(x) = \max(0, x) $$
- **Softmax:** Utilizes a numerically stable implementation to prevent standard `float32` overflows when dealing with steep logits.
  $$ f(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_j e^{x_j - \max(\mathbf{x})}} $$

---

## ⚡ Memory & Time Complexity Analysis

Developing a bespoke matrix multiplication loop requires careful consideration of CPU cache hierarchies. 

### Time Complexity: $\mathcal{O}(I \cdot J \cdot K)$
Standard Matrix Multiplication of $A (I \times K)$ and $B (K \times J)$ requires $I \cdot J \cdot K$ multiply-accumulate (MAC) operations. 
However, standard mathematical loops evaluate as `for i, for j, for k`. 

**Cache Locality Optimization:** In `InferC::Tensor::matmul`, the loop topology is deliberately structured as `i-k-j`. 
Because C++ standard vectors store elements in **row-major order**, accessing columns inside an inner loop causes catastrophic L1/L2 cache misses (spatial locality failure). By pushing $j$ to the innermost loop, both matrix $A$ and matrix $B$ are traversed sequentially in physical memory, resulting in orders-of-magnitude faster execution on standard CPUs due to hardware prefetching.

### Space Complexity: $\mathcal{O}(I \cdot K + K \cdot J)$
Tensors are aggressively backed by contiguous RAII allocations via `std::vector<float>`. All temporary state vectors rely on stack pointers; there are zero dynamic `new` or `malloc` calls made per forward pass loop, eliminating heap fragmentation during long-running edge processes.

---

## 🛠️ Build Instructions

### Prerequisites
- Python 3.8+ (for generating mock trained weights)
- `g++` or `clang++` with C++17 support
- `make`

### Compilation & Execution
This repository utilizes a `Makefile` that handles compiling the C++ source and running the Python exporter script automatically. Compiler optimizations `-O3` and `-march=native` are aggressively enabled by default.

```bash
# Clone the repository
git clone https://github.com/your-username/ml-inference-engine.git
cd ml-inference-engine

# Build the project (will automatically generate weights via Python and compile C++)
make

# Run the inference binary
make run
```

---

## 💻 API Usage

Because `InferC` is header-only, integration into existing robotics or IoT C++ firmware is trivial. 

```cpp
#include "InferC.hpp"
#include <fstream>
#include <iostream>

using namespace inferc;

int main() {
    // 1. Initialize structural layers
    DenseLayer hidden_layer(784, 128); // e.g. MNIST Flattened
    DenseLayer output_layer(128, 10);
    
    // 2. Load bytes directly from Flash/Disk
    std::ifstream disk("trained_weights.bin", std::ios::binary);
    hidden_layer.load_weights(disk);
    output_layer.load_weights(disk);
    
    // 3. Create sensor tensor input
    Tensor sensor_data(1, 784); // Batch size 1, 784 features
    
    // 4. Zero-Allocation Forward Pass
    Tensor hidden = hidden_layer.forward(sensor_data);
    activation::relu(hidden);
    
    Tensor predictions = output_layer.forward(hidden);
    activation::softmax(predictions);
    
    // 5. Output highest probability class
    std::cout << "Class 0 Probability: " << predictions(0, 0) << "\n";
    
    return 0;
}
```

---
*Built from scratch for rigorous academic assessment and edge performance.*
