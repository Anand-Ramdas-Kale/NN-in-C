# Compiler and flags
CC      := nvcc
CFLAGS  := -O3 -g
LDFLAGS := -lcublas -lm

# Directories
SRC_DIR := ../src
BUILD_DIR := ../build

# Files
SRC := $(wildcard $(SRC_DIR)/*.cu)
OBJ := $(SRC:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)
BIN := $(BUILD_DIR)/main

# Default target
all: $(BIN)

# Linking
$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compiling .cu to .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean
