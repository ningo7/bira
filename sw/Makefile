PYTHON ?= python3
HOST_CC ?= gcc
RISCV_CC ?= riscv64-unknown-elf-gcc
VERILATOR ?= verilator

SW_ROOT := $(abspath $(CURDIR))
RUNTIME_ROOT := $(SW_ROOT)/runtime
MODEL_ROOT := $(SW_ROOT)/models/bfsrcnn
BUILD_DIR ?= $(SW_ROOT)/build
BFSRCNN_BUILD ?= $(BUILD_DIR)/bfsrcnn
GENERATED_DIR ?= $(BFSRCNN_BUILD)/generated
FIXTURE_DIR ?= $(BFSRCNN_BUILD)/fixture
RISCV_DIR ?= $(BFSRCNN_BUILD)/riscv

FLOAT_CHECKPOINT ?= $(MODEL_ROOT)/outputs/x4/epoch_26_qat_shrink1w4_a8_exact.pth
INTEGER_CHECKPOINT ?= $(MODEL_ROOT)/tests/fixtures/bfsrcnn_integer_v4.pth
QUANTIZED_CHECKPOINT ?= $(BFSRCNN_BUILD)/quantized/bfsrcnn_integer.pth
CALIBRATION_DIR ?= $(MODEL_ROOT)/datasets/Set14
INPUT_IMAGE ?= $(MODEL_ROOT)/datasets/Set14/baboon.png
INPUT_HEIGHT ?= 32
INPUT_WIDTH ?= 32
INPUT_SCALE ?= 4
INPUT_TILE_SIZE ?= 24
INPUT_TILE_PAD ?= 4
INPUT_TILE_ROW ?= 1
INPUT_TILE_COLUMN ?= 1

CFLAGS := -std=c99 -O2 -Wall -Wextra -Werror -I$(RUNTIME_ROOT)/include
RISCV_CFLAGS := $(CFLAGS) -march=rv64gc -mabi=lp64d

.PHONY: test runtime-test layout-test simulation-test model-test \
	bfsrcnn-quantize bfsrcnn-generate bfsrcnn-fixture \
	bfsrcnn-verilator bfsrcnn-riscv bfsrcnn-all clean-generated

test: runtime-test layout-test simulation-test model-test

runtime-test:
	$(PYTHON) -m unittest \
		runtime.tests.test_bira_runtime \
		runtime.tests.test_bira_ops \
		runtime.tests.test_bira_inference -v

layout-test:
	$(PYTHON) -m unittest \
		models.common.deployment.tests.test_parameter_layout -v

simulation-test:
	$(PYTHON) -m unittest \
		simulation.tests.test_command_codec -v

model-test:
	$(PYTHON) -m models.bfsrcnn.network.model_info \
		--scale 4 --height $(INPUT_HEIGHT) --width $(INPUT_WIDTH)

bfsrcnn-quantize:
	mkdir -p $(dir $(QUANTIZED_CHECKPOINT))
	$(PYTHON) -m models.bfsrcnn.quantization.quantize \
		--weights-file $(FLOAT_CHECKPOINT) \
		--output-file $(QUANTIZED_CHECKPOINT) \
		--calibration-dir $(CALIBRATION_DIR) \
		--shrink3-weight-bits 8 \
		--expand-weight-bits 8

bfsrcnn-generate:
	$(PYTHON) -m models.bfsrcnn.deployment.export \
		--checkpoint $(INTEGER_CHECKPOINT) \
		--output-dir $(GENERATED_DIR) \
		--height $(INPUT_HEIGHT) \
		--width $(INPUT_WIDTH)

bfsrcnn-fixture:
	$(PYTHON) -m simulation.bfsrcnn_fixture \
		--checkpoint $(INTEGER_CHECKPOINT) \
		--generated-dir $(GENERATED_DIR) \
		--output-dir $(FIXTURE_DIR) \
		--input-image $(INPUT_IMAGE) \
		--input-scale $(INPUT_SCALE) \
		--tile-size $(INPUT_TILE_SIZE) \
		--tile-pad $(INPUT_TILE_PAD) \
		--tile-row $(INPUT_TILE_ROW) \
		--tile-column $(INPUT_TILE_COLUMN)

bfsrcnn-verilator: bfsrcnn-fixture
	$(MAKE) -C $(SW_ROOT)/../hw/sim/models/bfsrcnn run \
		VERILATOR=$(VERILATOR) \
		FIXTURE=$(abspath $(FIXTURE_DIR))

bfsrcnn-riscv: bfsrcnn-generate
	mkdir -p $(RISCV_DIR)
	$(RISCV_CC) $(RISCV_CFLAGS) -I$(GENERATED_DIR) \
		-c $(RUNTIME_ROOT)/src/bira_runtime.c \
		-o $(RISCV_DIR)/bira_runtime.o
	$(RISCV_CC) $(RISCV_CFLAGS) -I$(GENERATED_DIR) \
		-c $(RUNTIME_ROOT)/src/bira_inference.c \
		-o $(RISCV_DIR)/bira_inference_engine.o
	$(RISCV_CC) $(RISCV_CFLAGS) -I$(GENERATED_DIR) \
		-c $(RUNTIME_ROOT)/src/bira_planner.c \
		-o $(RISCV_DIR)/bira_planner.o
	$(RISCV_CC) $(RISCV_CFLAGS) -I$(GENERATED_DIR) \
		-c $(GENERATED_DIR)/bfsrcnn_bira_data.c \
		-o $(RISCV_DIR)/bfsrcnn_bira_data.o
	$(RISCV_CC) $(RISCV_CFLAGS) -I$(GENERATED_DIR) \
		-I$(SW_ROOT)/applications/bfsrcnn \
		-c $(SW_ROOT)/applications/bfsrcnn/bfsrcnn_inference.c \
		-o $(RISCV_DIR)/bfsrcnn_inference.o

bfsrcnn-all: test bfsrcnn-fixture simulation-test

clean-generated:
	@echo "generated files are under $(BFSRCNN_BUILD); remove that explicit directory when desired"
