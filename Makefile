# Compiler settings - standard C++17
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pedantic -Wno-unused-parameter

# Hardware optimization flags (optional but recommended for Edge/Inference)
# -march=native tells the compiler to optimize using the host CPU's instruction set (AVX, SSE)
OPTFLAGS = -march=native

# Directories
INCLUDES = -I./include
SRC_DIR = ./src
OBJ_DIR = ./obj
BIN_DIR = ./bin

# Target executable
TARGET = $(BIN_DIR)/infer_engine

# Sources and Objects
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

# Default target
all: directories $(TARGET) run_python

# Linking
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $^

# Compilation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDES) -c $< -o $@

# Utility targets
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

run_python:
	@echo "Generating dummy weights using Python..."
	python3 scripts/export_weights.py
	@mv model_weights.bin $(BIN_DIR)/

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
	rm -f model_weights.bin

# Run the inference engine
run: all
	@cd $(BIN_DIR) && ./infer_engine

.PHONY: all clean directories run run_python
