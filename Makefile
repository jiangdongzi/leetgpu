PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
NVCC ?= nvcc
CUDA_SO := cuda/vector_add.so
HELLO_BIN := build/hello_world
GEMM_BIN := build/gemm
LANE_LAYOUT_BENCH_BIN := build/lane_layout_shared_bench
SYSTEM_SITE ?= /opt/ml-stack/venv/lib/python3.10/site-packages

# CUDA 架构: RTX 5070 = Blackwell consumer = sm_120 (CUDA >= 12.8 才支持)
# 想编多架构通用包: CUDA_ARCH ?= -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120
CUDA_ARCH ?= -arch=sm_120

.PHONY: help venv install build hello gemm gemm-run lane-layout-bench lane-layout-bench-run test perf clean

help:
	@echo "Targets:"
	@echo "  make venv     - create virtual environment and install dependencies"
	@echo "  make build    - compile CUDA shared library"
	@echo "  make hello    - compile and run CUDA hello world"
	@echo "  make gemm     - compile cuda/gemm.cu (SGEMM benchmark, links cuBLAS)"
	@echo "  make gemm-run - compile then run the SGEMM benchmark"
	@echo "  make lane-layout-bench-run - benchmark shared-memory lane layouts"
	@echo "  make test     - run functional verification for PyTorch/CUDA/Triton"
	@echo "  make perf     - run a larger performance smoke test"
	@echo "  make clean    - remove generated artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	echo "$(SYSTEM_SITE)" > $(VENV)/lib/python3.10/site-packages/ml_stack_bridge.pth
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements/dev.txt

install: venv

build: $(CUDA_SO)

$(CUDA_SO): cuda/vector_add.cu
	$(NVCC) -O3 --shared -Xcompiler -fPIC -o $(CUDA_SO) cuda/vector_add.cu

hello: $(HELLO_BIN)
	./$(HELLO_BIN)

$(HELLO_BIN): cuda/hello_world.cu
	mkdir -p build
	$(NVCC) -O2 -o $(HELLO_BIN) cuda/hello_world.cu

gemm: $(GEMM_BIN)

gemm-run: $(GEMM_BIN)
	./$(GEMM_BIN)

$(GEMM_BIN): cuda/gemm.cu
	mkdir -p build
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH) -lineinfo -Xptxas -v \
	    -o $(GEMM_BIN) cuda/gemm.cu -lcublas

lane-layout-bench: $(LANE_LAYOUT_BENCH_BIN)

lane-layout-bench-run: $(LANE_LAYOUT_BENCH_BIN)
	./$(LANE_LAYOUT_BENCH_BIN)

$(LANE_LAYOUT_BENCH_BIN): cuda/lane_layout_shared_bench.cu
	mkdir -p build
	$(NVCC) -O3 -std=c++17 $(CUDA_ARCH) -lineinfo -Xptxas -v \
	    -o $(LANE_LAYOUT_BENCH_BIN) cuda/lane_layout_shared_bench.cu

test: build
	$(PYTHON) scripts/verify_vector_add.py --mode functional

perf: build
	$(PYTHON) scripts/verify_vector_add.py --mode performance

clean:
	rm -rf $(VENV) $(CUDA_SO) $(HELLO_BIN) $(GEMM_BIN) $(LANE_LAYOUT_BENCH_BIN) cuda/gemm .pytest_cache __pycache__
