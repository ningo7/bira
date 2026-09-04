VERILATOR ?= verilator
PYTHON ?= python3
SBT ?= sbt

MODEL_SIM_ROOT := $(abspath $(CURDIR))
SIM_ROOT := $(abspath $(MODEL_SIM_ROOT)/../..)
HW_ROOT := $(abspath $(SIM_ROOT)/..)
REPO_ROOT := $(abspath $(HW_ROOT)/..)
SW_ROOT := $(REPO_ROOT)/sw
RTL_DIR ?= $(HW_ROOT)/build/generated-rtl
RTL := $(RTL_DIR)/StandaloneTop.sv
BUILD_DIR ?= $(HW_ROOT)/build/verilator/bfsrcnn
FIXTURE ?= $(HW_ROOT)/src/test/resources/bira/bfsrcnn4
GENERATED_FIXTURE ?= $(SW_ROOT)/build/bfsrcnn/fixture
HARNESS ?= $(MODEL_SIM_ROOT)/bfsrcnn_trace_main.cpp
EXECUTABLE := $(BUILD_DIR)/VStandaloneTop
RTL_SOURCES := $(shell find $(HW_ROOT)/src/main/scala/bira -type f -name '*.scala')

.PHONY: rtl build run fixture profile

rtl: $(RTL)

$(RTL): $(RTL_SOURCES)
	cd $(HW_ROOT) && $(SBT) \
		"runMain bira.GenStandaloneTop $(RTL_DIR)"

build: $(RTL)
	mkdir -p $(BUILD_DIR)
	$(VERILATOR) --cc --exe --build -j 0 \
		--top-module StandaloneTop \
		--Mdir $(BUILD_DIR) \
		$(RTL) \
		$(HARNESS) \
		-CFLAGS "-O3 -std=c++17"

run: build
	$(EXECUTABLE) $(FIXTURE)

fixture:
	$(MAKE) -C $(SW_ROOT) bfsrcnn-fixture PYTHON=$(PYTHON)

profile: build fixture
	$(EXECUTABLE) $(GENERATED_FIXTURE)
