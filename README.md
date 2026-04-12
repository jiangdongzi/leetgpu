# LeetGPU Init

这个仓库已经按 `PyTorch`、`CUDA`、`Triton` 三种方式初始化了 LeetGPU 第一题 `Vector Addition`。

## 目录

- `pytorch/vector_add.py`: PyTorch 实现
- `triton_impl/vector_add.py`: Triton kernel 实现
- `cuda/vector_add.cu`: CUDA kernel 实现
- `cuda_impl/vector_add.py`: CUDA `.so` 加载器
- `scripts/verify_vector_add.py`: 统一验证脚本
- `requirements/dev.txt`: Python 依赖
- `Makefile`: 环境、编译、测试入口

## 使用

```bash
make build
make test
make perf
```

如果当前 Python 环境还没有安装 `torch`、`triton`，可以执行：

```bash
make venv
source .venv/bin/activate
make test
```
