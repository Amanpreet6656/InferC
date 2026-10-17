# InferC: C++17 Machine Learning Inference Engine

This repository contains the source code for InferC, a header-only C++ library engineered for neural network inference on edge computing devices. The system operates entirely without external dependencies (such as Eigen, OpenCV, or TensorFlow). It implements core tensor mathematics, cache-optimized matrix multiplications, and standard neural network operations natively via the C++ standard library, bridging the gap between high-level Python training environments and bare-metal deployment targets.

## Mathematical Architecture

The engine simulates standard Feed-Forward neural network operations explicitly. 

### Fully Connected (Dense) Layer
For a layer $L$, given an input matrix $X \in \mathbb{R}^{B \times I}$ (where $B$ represents batch size and $I$ represents input features), weight matrix $W \in \mathbb{R}^{I \times O}$, and bias vector $B \in \mathbb{R}^{1 \times O}$, the forward transformation is defined as:
$$Y = X \cdot W + B$$

### Non-Linear Activation Functions
*   **ReLU (Rectified Linear Unit):** Applies a piecewise linear function over the tensor $T$:
    $$f(x) = \max(0, x)$$
*   **Softmax:** Implemented with numerical stability adjustments to mitigate standard `float32` overflow when processing steep logits:
    $$f(x_i) = \frac{e^{x_i - \max(\mathbf{x})}}{\sum_j e^{x_j - \max(\mathbf{x})}}$$

## Computational Complexity and Memory Architecture

### Time Complexity: $\mathcal{O}(I \cdot J \cdot K)$
Standard matrix multiplication of $A (I \times K)$ and $B (K \times J)$ requires $I \cdot J \cdot K$ multiply-accumulate (MAC) operations. 

**Cache Locality Optimization:** The iteration topology in `InferC::Tensor::matmul` is explicitly structured sequentially as $i \rightarrow k \rightarrow j$. Because `std::vector` allocates memory in row-major order, traversing columns within an inner loop degrades L1/L2 cache spatial locality. Structuring $j$ as the innermost loop ensures matrices $A$ and $B$ are accessed sequentially in contiguous physical memory, leveraging CPU hardware prefetching mechanisms to optimize execution speed.

### Space Complexity: $\mathcal{O}(I \cdot K + K \cdot J)$
Tensors are managed via contiguous RAII allocations utilizing `std::vector<float>`. Temporary state vectors are resolved via the stack pointer. The forward pass executes without dynamic heap allocations (`new` or `malloc`), eliminating heap fragmentation during sustained edge deployment.

## Build Instructions

### Prerequisites
*   Python 3.8+ (utilized for weight generation scripts)
*   `g++` or `clang++` (must support the C++17 standard)
*   `make`

### Compilation & Execution
Compilation is managed via the included `Makefile`, which handles both the C++ source and the Python exporter script. The system compiles utilizing `-O3` and `-march=native` compiler optimizations by default.

```bash
# Clone the repository
git clone [https://github.com/your-username/ml-inference-engine.git](https://github.com/your-username/ml-inference-engine.git)
cd ml-inference-engine

# Execute the build sequence (generates weights via Python and compiles the C++ binary)
make

# Execute the compiled binary
make run
