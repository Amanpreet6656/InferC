import struct
import random

def write_layer(file, in_dim, out_dim):
    """
    Writes a dense layer's dimensions, weights, and biases to a binary file.
    Weights are initialized with random float32 values.
    """
    # Write dimensions as 32-bit integers
    file.write(struct.pack('ii', in_dim, out_dim))
    
    # Generate and write dummy weights (in_dim * out_dim)
    # Using random standard normal-like distribution (dummy values)
    print(f"Exporting Layer ({in_dim}x{out_dim})...")
    for _ in range(in_dim * out_dim):
        weight = random.uniform(-0.5, 0.5)
        file.write(struct.pack('f', weight))
        
    # Generate and write dummy biases (out_dim)
    for _ in range(out_dim):
        bias = random.uniform(-0.1, 0.1)
        file.write(struct.pack('f', bias))

def export_model(filename):
    """
    Exports a 2-layer Neural Network model to a binary file.
    Architecture: 4 -> 16 -> 3
    """
    print(f"Initializing export to {filename}...")
    
    with open(filename, 'wb') as f:
        # Layer 1: Input(4) -> Hidden(16)
        write_layer(f, 4, 16)
        
        # Layer 2: Hidden(16) -> Output(3)
        write_layer(f, 16, 3)
        
    print(f"Successfully exported raw weights to {filename}.")

if __name__ == "__main__":
    # Ensure reproducibility for testing
    random.seed(42)
    export_model("model_weights.bin")
