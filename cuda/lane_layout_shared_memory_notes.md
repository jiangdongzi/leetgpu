# GEMM Lane Layout 与 Shared Memory 访问笔记

本文整理 `cuda/gemm.cu` 中 warp 内 lane 排布对 shared memory `FLOAT4` 读取性能的影响，以及对应的最小压测验证。

## 背景

`cuda/gemm.cu` 中当前使用的 lane 排布为：

```cpp
int lanex = (laneId & 15) >> 1;
int laney = ((laneId >> 4) << 1) + (laneId & 1);
```

对比的 naive 排布为：

```cpp
int lanex = laneId % 8;
int laney = laneId / 8;
```

更高效的替代写法可以把 `%8` 和 `/8` 写成位运算：

```cpp
int lanex = laneId & 7;
int laney = laneId >> 3;
```

但这只是在表达 naive 排布，性能差异的核心不在 `%`/`/` 本身，而在 lane 到 `(lanex, laney)` 的映射方式改变了 shared memory 访问分布。

## 两种 Lane 排布

当前 swizzled 排布下，前 8 个 lane 的坐标是：

```text
lane : 0 1 2 3 4 5 6 7
lanex: 0 0 1 1 2 2 3 3
laney: 0 1 0 1 0 1 0 1
```

naive 排布下，前 8 个 lane 的坐标是：

```text
lane : 0 1 2 3 4 5 6 7
lanex: 0 1 2 3 4 5 6 7
laney: 0 0 0 0 0 0 0 0
```

也就是说，当前排布让相邻 lane 成对共享 `lanex`，而 naive 排布让相邻 lane 共享 `laney`。

## 影响的代码位置

`lanex/laney` 会生成 `tCx/tCy`：

```cpp
int tCx = warpx * THREAD_C_WARP_X + lanex;
int tCy = warpy * THREAD_C_WARP_Y + laney;
```

随后直接影响 inner loop 中的 shared memory 读取：

```cpp
for (int tm = 0; tm < THREAD_C_Y_TILE_SIZE >> 2; tm ++) {
    int r = (tCy + tm * THREAD_C_LAYOUT_Y) << 2;
    FLOAT4(tCsA[tk & 1][tm << 2]) = FLOAT4(sA[bufferId][tk][r ^ (tk << 2)]);
}

for (int tn = 0; tn < THREAD_C_X_TILE_SIZE >> 2; tn ++) {
    int c = (tCx + tn * THREAD_C_LAYOUT_X) << 2;
    FLOAT4(tCsB[tk & 1][tn << 2]) = FLOAT4(sB[bufferId][tk][c]);
}
```

其中：

- `A` 的 shared load 地址主要由 `tCy/laney` 决定。
- `B` 的 shared load 地址主要由 `tCx/lanex` 决定。
- 每个 `FLOAT4` 是一次 16B shared load，也就是 `ld.shared.v4.u32` / `LDS.128` 级别的访问。

## Warp 内 Lane 是否完全对等

从 SIMT 执行语义看，同一个 warp 内所有 lane 是对等的：它们执行同一条指令。

但从内存系统看，lane id 的相对位置会影响：

- shared memory bank 分布；
- 同一次 memory transaction 内的 bank conflict；
- 同地址访问是否能 broadcast/multicast；
- `LDS.128` 这类向量化 shared load 被拆成哪些 transaction。

因此，`lane0` 和 `lane1` 都在同一个 warp，`lane0` 和 `lane8` 也在同一个 warp，但它们在一次 `FLOAT4` shared load 中不一定处在同一个 shared-memory transaction 里。

## Shared Memory Transaction 与 Broadcast

NVIDIA 文档中的核心规则是：

- shared memory 被划分成多个 bank；
- 一个 memory request 中，如果地址落在不同 bank，可以并行服务；
- 如果同一个 memory request 中有多个不同地址落在同一个 bank，会产生 bank conflict，并被拆分/串行；
- 如果多个 lane 访问同一个地址，可以在对应 transaction 内 broadcast/multicast，不构成普通意义上的 bank conflict。

关键限定是：broadcast/multicast 是发生在同一次 shared-memory transaction/request 内，而不是“同一个 warp 内任意两个 lane 读同一个地址都一定合并”。

在 `ld.shared.v4.u32` 场景中：

```text
32 lanes * 16B = 512B
```

硬件通常会按更小的 transaction 粒度处理，例如可以近似理解为：

```text
transaction 0: lane0  - lane7
transaction 1: lane8  - lane15
transaction 2: lane16 - lane23
transaction 3: lane24 - lane31
```

所以：

```text
lane0 和 lane1/2/3/4/5/6/7
```

如果读同一个 shared 地址，更容易在同一个 transaction 内 broadcast/multicast。

而：

```text
lane0 和 lane8
```

即使读同一个 shared 地址，也很可能已经处在不同 transaction 中，不能作为同一次 transaction 内的 broadcast 合并。

## 为什么当前排布更利于 B

当前 swizzled 排布下，B 的复用发生在相邻 lane：

```text
lane0,1   -> same lanex -> same B float4
lane2,3   -> same lanex -> same B float4
lane4,5   -> same lanex -> same B float4
lane6,7   -> same lanex -> same B float4
```

这些 lane 更可能落在同一个 `ld.shared.v4.u32` transaction 里，因此同地址读取可以更有效地 broadcast/multicast，B 侧 shared load 更快。

naive 排布下，B 的复用关系变成：

```text
lane0,8,16,24 -> same lanex -> same B float4
lane1,9,17,25 -> same lanex -> same B float4
lane2,10,18,26 -> same lanex -> same B float4
lane3,11,19,27 -> same lanex -> same B float4
```

这些 lane 虽然仍在同一个 warp 内，但被分散到了不同 transaction 中，因此不能高效地在同一次 shared-memory transaction 内合并。

## 简化性能模型

我们讨论过一个简化模型：暂时假设一次无冲突 shared memory 读占 1 个周期。

在这个模型里：

```text
swizzled:
  两次 FLOAT4(A): 4 个单位
  两次 FLOAT4(B): 4 个单位
  合计: 8 个单位

naive:
  两次 FLOAT4(A): 2 个单位
  两次 FLOAT4(B): 8 个单位
  合计: 10 个单位
```

这个模型表达了一个重要方向：当前排布可能没有让 A 侧最优，但显著改善了 B 侧；由于 inner loop 每个 `tk` 都要读两次 A 和两次 B，B 侧收益更大，最终总路径更快。

需要注意的是，这只是 bank-conflict 等效开销模型，不是硬件真实周期数。真实周期还会受到 transaction 拆分、issue、scoreboard、流水重叠、编译器生成指令等因素影响。

## 最小压测 Demo

为了验证这个判断，新增了：

```text
cuda/lane_layout_shared_bench.cu
```

它保留 `gemm.cu` 中同样的地址公式，分别测试：

```text
A-only:  每轮只执行两次 FLOAT4(A)
B-only:  每轮只执行两次 FLOAT4(B)
A+B:     每轮执行两次 FLOAT4(A) 和两次 FLOAT4(B)
```

编译运行：

```bash
make lane-layout-bench-run
```

也可以手动传参数：

```bash
./build/lane_layout_shared_bench [iters] [blocks] [repeats]
```

例如：

```bash
./build/lane_layout_shared_bench 131072 1 9
```

## 压测结果

在 RTX 5070 上，默认参数的一次结果约为：

```text
B-only: naive / swizzled = 1.546x
A+B   : naive / swizzled = 1.237x
A-only: naive / swizzled = 1.007x
```

单 block 参数的一次结果约为：

```text
B-only: naive / swizzled = 1.167x
A+B   : naive / swizzled = 1.154x
A-only: naive / swizzled = 0.978x
```

这说明：

- naive 排布下，B-only 明显更慢；
- A+B 总路径也明显更慢；
- A-only 在这张卡上基本持平，甚至 naive 略快一点；
- 因此真实硬件上，A 侧差异没有简化模型里那么强，B 侧才是主要收益来源。

## 最终结论

当前 `gemm.cu` 的 lane 排布更快，核心原因不是位运算比 `%`/`/` 快，而是它改变了同一个 warp 内 lane 到输出坐标的映射，使 B 的 shared-memory `FLOAT4` 读取复用发生在相邻 lane 中。

更准确地说：

```text
同一个 warp 内任意两个 lane 读同一个 shared 地址，不一定都能一样高效地合并。
同一个 shared-memory transaction 内的同地址读取，才更容易 broadcast/multicast。
```

因此，`lane0` 和 `lane8` 即使读同一个 shared 地址，也可能因为不在同一个 `LDS.128` transaction 中而不能合并；而 `lane0` 和 `lane1` 这类相邻 lane 的同地址访问，更容易从 broadcast/multicast 中受益。

