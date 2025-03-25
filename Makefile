debug ?= 0
NAME := siboehm_sgemm
SRC_DIR := src
BUILD_DIR := build
INCLUDE_DIR := ./include
LIB_DIR := lib
TESTS_DIR := tests
BIN_DIR := bin

# Generate paths to all object files
OBJS := $(patsubst %.cu, %.o, $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(LIB_DIR)/**/*.c))

# Compiler settings
CC := nvcc
CFLAGS :=
LDFLAGS := -lm -I$(INCLUDE_DIR)

# Targets

# Build executable

$(NAME): $(OBJS)
	$(CC) $(CFLAGS) -o $(BIN_DIR)/$@ $(patsubst %, build/%, $(OBJS)) $(LDFLAGS)

$(OBJS): dir
	@mkdir -p $(BUILD_DIR)/$(@D)
	@$(CC) $(CFLAGS) -o $(BUILD_DIR)/$@ -c $*.cu $(LDFLAGS)

setup:
# Put dependencies installs here if needed

dir:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: dir setup clean