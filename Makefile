PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
NVCC ?= nvcc
CUDA_SO := cuda/vector_add.so
SYSTEM_SITE ?= /opt/ml-stack/venv/lib/python3.10/site-packages

.PHONY: help venv install build test perf clean

help:
	@echo "Targets:"
	@echo "  make venv    - create virtual environment and install dependencies"
	@echo "  make build   - compile CUDA shared library"
	@echo "  make test    - run functional verification for PyTorch/CUDA/Triton"
	@echo "  make perf    - run a larger performance smoke test"
	@echo "  make clean   - remove generated artifacts"

venv:
	$(PYTHON) -m venv $(VENV)
	echo "$(SYSTEM_SITE)" > $(VENV)/lib/python3.10/site-packages/ml_stack_bridge.pth
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements/dev.txt

install: venv

build: $(CUDA_SO)

$(CUDA_SO): cuda/vector_add.cu
	$(NVCC) -O3 --shared -Xcompiler -fPIC -o $(CUDA_SO) cuda/vector_add.cu

test: build
	$(PYTHON) scripts/verify_vector_add.py --mode functional

perf: build
	$(PYTHON) scripts/verify_vector_add.py --mode performance

clean:
	rm -rf $(VENV) $(CUDA_SO) .pytest_cache __pycache__
