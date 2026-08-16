BIRA_ROOT := $(abspath ../../..)
SW_ROOT := $(BIRA_ROOT)/sw
CHIPYARD_ROOT := $(abspath $(BIRA_ROOT)/../..)
RISCV ?= $(CHIPYARD_ROOT)/.conda-env/riscv-tools
CC := $(RISCV)/bin/riscv64-unknown-elf-gcc
OBJDUMP := $(RISCV)/bin/riscv64-unknown-elf-objdump
COMMON := $(CHIPYARD_ROOT)/toolchains/riscv-tools/riscv-tests/benchmarks/common
RISCV_TEST_ENV := $(CHIPYARD_ROOT)/toolchains/riscv-tools/riscv-tests/env
RUNTIME_INCLUDE := $(SW_ROOT)/runtime/include
BUILD_DIR := $(SW_ROOT)/build/chipyard-baremetal
SIMULATOR ?= $(CHIPYARD_ROOT)/sims/verilator/simulator-chipyard.harness-BiRaRocketConfig

TESTS := bira_status bira_dma_loopback bira_dense_1x1 bira_binary_1x1
ELFS := $(addprefix $(BUILD_DIR)/,$(addsuffix .riscv,$(TESTS)))
RUN_TARGETS := $(addprefix run-,$(TESTS))
BFSRCNN_DIR := $(SW_ROOT)/build/bfsrcnn
BFSRCNN_GENERATED := $(BFSRCNN_DIR)/generated
BFSRCNN_FIXTURE := $(BFSRCNN_DIR)/fixture
BFSRCNN_ELF := $(BUILD_DIR)/bira_bfsrcnn.riscv
BFSRCNN_SOURCES := \
	$(SW_ROOT)/runtime/src/bira_runtime.c \
	$(SW_ROOT)/runtime/src/bira_inference.c \
	$(SW_ROOT)/runtime/src/bira_planner.c \
	$(BFSRCNN_GENERATED)/bfsrcnn_bira_data.c \
	$(SW_ROOT)/applications/bfsrcnn/bfsrcnn_inference.c

CFLAGS := \
	-march=rv64gc -mabi=lp64d -mcmodel=medany \
	-std=gnu99 -O2 -fno-common -fno-builtin \
	-fno-tree-loop-distribute-patterns -Wall -Wextra \
	-I$(RUNTIME_INCLUDE) -I$(COMMON) -I$(RISCV_TEST_ENV)
LDFLAGS := \
	-static -nostdlib -nostartfiles \
	-T$(COMMON)/test.ld -lgcc

.PHONY: all clean disassemble run run-bfsrcnn $(RUN_TARGETS)

all: $(ELFS)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.riscv: %.c bira_test_common.c bira_test_common.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) \
		$< bira_test_common.c $(COMMON)/syscalls.c $(COMMON)/crt.S \
		$(LDFLAGS) -o $@

$(BFSRCNN_ELF): bira_bfsrcnn.c bira_bfsrcnn_fixture.S \
		$(BFSRCNN_SOURCES) \
		$(BFSRCNN_FIXTURE)/input.bin $(BFSRCNN_FIXTURE)/expected.bin \
		| $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(BFSRCNN_GENERATED) \
		-I$(SW_ROOT)/applications/bfsrcnn \
		bira_bfsrcnn.c bira_bfsrcnn_fixture.S \
		$(BFSRCNN_SOURCES) \
		$(COMMON)/syscalls.c $(COMMON)/crt.S \
		$(LDFLAGS) -o $@

$(BUILD_DIR)/%.dump: $(BUILD_DIR)/%.riscv
	$(OBJDUMP) -d -S $< > $@

disassemble: $(addprefix $(BUILD_DIR)/,$(addsuffix .dump,$(TESTS)))

run: $(RUN_TARGETS)

$(RUN_TARGETS): run-%: $(BUILD_DIR)/%.riscv
	$(SIMULATOR) $<

run-bfsrcnn: $(BFSRCNN_ELF)
	$(SIMULATOR) $<

clean:
	rm -f $(ELFS) $(BFSRCNN_ELF) \
		$(addprefix $(BUILD_DIR)/,$(addsuffix .dump,$(TESTS)))
