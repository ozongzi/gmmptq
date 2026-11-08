# M1：tiny 家族的统一规格

行号针对抽取时的 `kernel/kernel.asc`（6122 行）。服务点 3 / 5 / 6，即分数缺口最大的三点（估分 26 / 32 / 31）。

## 簇 A —— 标量 tiny 行/专家核（8 个成员）

`RowParallelTiny` 1005、`E1FixedUbRowTiny` 1051、`BudgetStridedRowTiny` 1090、`SearchTinyE1STDev3400000` 4955、`SearchTinyExpert3500000` 4993、`SearchTinyExpertReuse6300000` 5044、`SearchTinyExpertOutput6400000` 5106、`SearchTinyRowSTDev3400000` 5200。

### 八者逐字节相同的部分

`KERNEL_TASK_TYPE_DEFAULT(AIV_ONLY)` 那一行；5 字段 `GeneralVectorDims` 位解包 `{(p&3)+1, ((p>>7)&7)+1, ((p>>2)&31)+1, ((p>>10)&127)+1, (p>>17)&1}`；5 个 `DataCachePreload` 建议性读；6 个 `SetGlobalBuffer`；内层标量 `sum += int8*int8` 循环；裸 `__gm__` 指针读 scale 时 `sizeof(ScaleT)==4 ? float : bits<<16` 的分叉；`float(sum)*scale*row` 的乘法顺序。

### 差异表

| 成员 | grid | owner 解析 | `n` | 存储 | UB | 其他 |
| --- | --- | --- | --- | --- | --- | --- |
| `RowParallelTiny` | `M` | 正向扫描，break | dims | 向量 `Cast`+`DataCopyPad` | `f@0`,`out@128` | preload gl；`InitSocState`；末尾 `PIPE_ALL` |
| `E1FixedUbRowTiny` | `M` | 常量 `0` | dims | 同上 | 同上 | 无 gl preload |
| `BudgetStridedRowTiny` | `min(M,2c)` | 正向扫描 | dims | 同上，逐行 | `slot*192` | 逐 slot `event_t`；`m += blocks` 跨步 |
| `SearchTinyE1STDev` | `M` | 常量 `0` | dims | `WriteGmByPassDCache` 16/32 位 + `DataSyncBarrier` | 无 | 无 `InitSocState` |
| `SearchTinyRowSTDev` | `M` | **双向**扫描 (5223-5236) | **强制 1** (5204) | 标量 bypass | 无 | xScale 在 gl **之前**读 (5220) |
| `SearchTinyExpert3500000` | `E` | `blockIdx` | dims | 整个专家一次 `DataCopyPad` | `f@0`,`out@128` | 行循环在核内 |
| `SearchTinyExpertReuse` | `E` | `blockIdx` | 2 | 同上 | 同上 | 2×2 的 w/ws 提到寄存器 (5076-5087) |
| `SearchTinyExpertOutput` | `E` | `blockIdx` | 2 | 打包 `uint64` 标量写，`begin` 为奇数时拆分 (5184-5193) | 无 | 4 行寄存器快照 (5150-5169) |

### 生成期参数（全部生成期解析，不留 C++ 模板；每种配置直接生成 fp16 与 bf16 两个具体核）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `grid` | `enum {PerRow, PerExpert, StridedRow}` | `GetBlockIdx()` 索引什么 |
| `owner` | `enum {Const0, Fwd, Bidir, BlockIsExpert}` | 专家归属解析 |
| `n_fixed` | `Option<u32>` | `Some(1)`=RowSTDev；`Some(2)`=Reuse/Output |
| `k_fixed` | `Option<u32>` | `Some(2)` ⇒ 生成展开的 2×2 body |
| `hoist_weights` | `bool` | 把 `w`/`ws` 提到行循环外（需要 `k_fixed` 与 `n_fixed`） |
| `row_snapshot` | `u32`（0=关，4=Output） | 预读 N 行 x/xs 进寄存器 |
| `store` | `enum {VecCopyPad, ScalarBypass, ScalarPackedWords}` | |
| `ub_slot_stride` | `Option<u32>` | `Some(192)` 给 Budget |
| `preload_grouplist` | `bool` | |
| `init_soc_state`, `final_pipe_all` | `bool` | 仅当 body 碰 UB/向量时需要 |
| `bf` | `bool` | 选 `(float,half)` 还是 `(bfloat16_t,bfloat16_t)`，以及 `GvEncode<BF>` |

约束：`VecCopyPad` 要求 `init_soc_state && final_pipe_all && ub_slot_stride.is_some()`；`ScalarBypass`/`ScalarPackedWords` 两者都不需要，但**必须**生成 `DataSyncBarrier<MemDsbT::DDR>`。

## 簇 B —— `tiny_capacity_unrolled` ↔ `tiny_capacity_contract`

近乎完全重复的一对。**逐字节相同**：`Load<Cap>`（5256/5419）、`Store<Cap>`（5271/5434）、`F`（5284/5447）、`S`（5285/5448）；`Rows` body 的三重 `#pragma unroll` 加 `XCap/WCap/SCap` 断言；5 个 `DataCachePreload`；dims 打包 `(M-1)|((K-1)<<2)|((N-1)<<5)|((E-1)<<7)|(gt<<14)`；`Select` 阶梯的形状，含 `TINY_CAPACITY_CPU` 的 `#ifdef` 与末尾的 `throw std::invalid_argument`。

| 方面 | unrolled | contract |
| --- | --- | --- |
| 行容量 | 推导 `MC=min(4,8/KC,8/NC)` (5288) | 显式 `RC` 模板参，`MC=RC` (5472) |
| 维度恢复 | 精确 `(dims&…)+1` (5313) | `DecodeCapacity<C>` 分桶+低位技巧 (5521-5525) |
| owner 解析 | 核内串行循环，每个 `Load<8>` 地址依赖前一个 (5321-5324) | `BatchOwners<RC>`：7 个**互不依赖**的 `Load<4>` + 前缀和 + 无分支 `+= (v<=row)` (5495-5519) |
| `E==1` | 独立 `Single=true` 实例 (5294, 5326-5327) | 无；`owners==0` |
| x / xs 加载 | 在 `Rows` 内 (5292) | 提到 owner 解析之前 (5536) |
| xs 打包 | 临时 `sx0/sx1` | `RowScales<RC>` + `RC==4` 特化 (5450-5459) |
| 输出打包 | 临时 `out0/out1` | `Output<Cells>` + `Cells==8` 特化 (5460-5469) |
| gl preload | `if(e>1)` (5315) | 无条件 (5530) |
| 实例数 | 9 个 `(KC,NC)` | 20 个 `(RC,KC,NC)` |
| host 闸门 | 无额外 | `2<=E<=8` 且 gl 完整落在一个 64B 行内 (5876-5877) |

**`contract` 在共同定义域上严格优于 `unrolled`**（独立 GL 加载、x 提前加载）。统一后的生成期参数：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `bf` | `bool` | scale/out dtype，生成 `GvEncode<BF>` |
| `rc` | `Option<u32>` ∈{1,2,4} | 行容量；`None` ⇒ 推导 `min(4,8/kc,8/nc)` |
| `kc`, `nc` | `u32` ∈{1,2,4,8} / {1,2,4} | K/N 容量分桶 |
| `dim_decode` | `enum {Exact, Bucketed}` | |
| `owners` | `enum {None, Sequential, Batched{max_e}}` | `Batched` 仅在 `max_e<=8` 有效 |
| `hoist_x` | `bool` | |
| `full` | `bool` | `k==KC && n==NC` 的快速实例——**两个都生成**，运行时选，与现状一致 |

## 必须保活的手调痕迹

| # | 行 | 内容 |
| --- | --- | --- |
| H4 | 1186-1188, 1536-1541, 3499-3509 | int8 配对拆分：`ShiftLeft(x,8)` 再 `ShiftRight(x,8)` 取偶车道，裸 `ShiftRight(x,8)` 取奇车道——从一个 int16 里符号扩展地抽出两个 int8，不用 `GatherMask` |
| H5 | 1210-1211, 1554, 3515-3537 | 乘积**绝不**在 int16 里累加（两个 16384 会溢出）。改为把两个 int16 乘积重打包进两个 int32 车道，用 `Add` 的重叠例外 `dst==src1, dstRepStride=src1RepStride=0` 归约——因为没有 int32 求和 intrinsic |
| H6 | 1215, 1217, 1454-1455 | `WholeReduceSum` 的 `dstRepStride` 以**元素**计而 `srcRepStride` 以 **32B 块**计。所有相关缓冲尺寸都依赖这一点，弄错会静默覆盖 `scales` |
| H9 | 1294-1295, 1362-1363, 5302-5303 | `#pragma unroll for(k=0;k<8;++k) if(k<kCount)`——固定循环次数配谓词，展开是静态的而 `K` 保持运行时量 |
| H10 | 1305-1310 | 每个 32 位 `SetValue` 打包两个 fp16/bf16 输出，末尾奇数元素用合法的 16 位存储 |
| H11 | 5256-5283, 5419-5446 | `Load<Cap>`/`Store<Cap>`：递归的 2 的幂字节切分器，用可得的最宽自然对齐访问并退化到半宽。`Store` 绕过 D-cache |
| H12 | 5521-5525 | `DecodeCapacity<C>`：按 `C/2 + 1 + (enc & (C/2-1))` 从分桶里恢复精确维度。**只在自己的桶内有效** |
| H13 | 5495-5519 | `BatchOwners`：7 个无条件互不依赖的 `Load<4>` + 前缀和 + 无分支 `owner += (v<=row)`，刻意做到没有任何 GM 读地址依赖于前一次 GM 读的结果 |
| H14 | 5150-5193 | `SearchTinyExpertOutput` 的 4 行寄存器快照与 `begin&1` 奇偏移存储拆分 |
| H15 | 5223-5236 | `SearchTinyRowSTDev` 的双向专家扫描：上半行号反向扫，期望扫描长度减半 |
| H16 | 5220-5221 | xScale 在 groupList **之前**读，让它的 cache 填充与 owner 搜索重叠。同思路见 1515-1524、3575-3594 |
| H20 | 1035/1047, 5022/5037 | `InitSocState()` + 末尾 `PipeBarrier<PIPE_ALL>` 手工替代 `TPipe` 构造/析构。少任一个都会弄坏不带 `TPipe` 却用 UB 的核 |

## 正确性雷区（生成器必须变成显式断言）

| # | 行 | 风险 |
| --- | --- | --- |
| L4 | 5521-5525 + 5546-5728 | `DecodeCapacity<2>` **无条件返回 2**，`DecodeCapacity<4>` 只返回 3 或 4。所以 `RC=2` 只能在 `M==2` 时进入、`RC=4` 只能在 `M∈{3,4}` 时进入。今天成立**仅仅**因为 `Select` 阶梯的 `m<=1/m<=2/m<=4` 分桶恰好吻合。重排或裁剪阶梯会静默越界写 `y` 2–4 字节。`KC`/`NC` 同理 |
| L5 | 5288, 5290 | `unrolled` 的 `MC = min(4, 8/KC, 8/NC)` 可能**小于**运行时 `m`；`>= MC` 的行从不计算，但 `bytes = m*n*2` 照样存（5309）⇒ `y` 里出现零。今天安全只因 host 闸门恰好蕴含 `M<=MC`。**把 `M<=MC` 写成显式生成期断言** |
| L6 | 5290, 5474 | `static_assert(XCap<=8 && WCap<=8 && SCap<=8 && MC*NC<=8)`——整族被「每操作数 8 字节」限死。`N*scale_sz<=8` 意味着 fp16（`scale_sz=4`）封顶 `N<=2` 而 bf16 封顶 `N<=4`；两个 `Select` 阶梯用 `if constexpr(BF)` 编码了这个不对称（5354, 5564） |
| L7 | 5495-5519 | `BatchOwners` 最多读 7 个边界 ⇒ `E<=8` 是硬上限。`E=9` 会静默把每行都判给 ≤7 号专家 |
| L8 | 1350 | `TinyShortReduction` 的 `uint32_t ends[16]` ⇒ `E<=16`。13312 字节固定 UB 还要求 `M*K<=512`、`E*K*N<=8192`、`4M<=256`、`E*N*4<=4096`、`(E-1)*8<=128`、`2*M*N<=128`，在 `K=8,E=16,N=64,M=64` 处全部恰好吃满 |
| L9 | 1237-1241 | `TinyCacheReduction` 把 `m` 塞 4 位（`M<=16`）、`n` 5 位、`k` 3 位、`e` **3 位**（`E<=8`）。host 只守了 `E==1`（5937），`E>8` 会回绕 |
| L10 | 1008-1009, 1320, 5313, 5528 | 同一组 dims 有 **4 套不同位布局**。簇 A：`m`2/`n`5/`k`3/`e`7；`TinyShortReduction`：`m`6/`n`6/`k`3/`e`4；`TinyCacheReduction`：`m`4/`n`5/`k`3/`e`3；TinyCapacity：`m`2/`k`3/`n`2/`e`7。**生成器只生成一套布局和一套解码器**，各核的字段宽度从此变成容量断言而非巧合 |
| L14 | 5858, 5949 | `E<=M` 出现在三处守卫里作为「按专家并行」的前提，它**不被任何东西蕴含**——点 5 就是 `E=5 > M=2`。任何 PerExpert grid 必须保留这个测试。反向地，`SearchTinyExpert*` 的 `end = owner+1<s.e ? gl[owner] : s.m`（5015, 5066, 5128）假定**隐含的最后端点等于 M**，这意味着 counts 之和小于 M 的列表会把尾部静默判给最后一个专家 |
| L15 | 1036-1037, 5023-5024 | `LocalTensor<float> f(VECCALC, 0, 32)` 加 `out` 在字节 128 ⇒ 共 32 个 float ⇒ `SearchTinyExpert*` 要求 `M*N<=32`、逐行核要求 `N<=32`。`BudgetStridedRowTiny` 的 `slot*192`（1123）假定 slot 总量不超过约 1000 字节。**这些都不是 `static_assert`，是静默的** |
| L16 | 5854 | `scalarOutputWindow = (y&127) + 2*M*N <= 32` 依赖 `y` 的**运行时地址**。64B 对齐分配下 `(y&127)` 是 0 或 64，所以这个谓词每次运行像抛硬币。**每个标量存储核都必须有可达的向量存储孪生核**，生成器不能假定某个形状有稳定路由 |

## 死代码（M1 范围内，直接删）

- `Pow2Below` 539-543、`Pow2AtLeast` 546-550
- `vec::LANES` 554、`vec::SCRATCH_BYTES` 565、`vec::BLOCK_ELEMS` 561（唯一用户是同样死掉的 `BuildVdotTiling` 910-932）
- `outer::BUDGET_ELEMS` 621
- `run_kernel` 5952-5955 的 `SearchTinyExpertOutput6400000` 内层分支：守卫集与 5858-5863 **完全相同**而后者先触发，不可达
- 5976-5982 的 `BudgetStridedRowTiny` 要求 `M > 2*availableCoreNum` 且 `M<=4`，即 `availableCoreNum<=1`。判题报约 20，实际死亡——是否保留作退化核数的安全网自行决定，但不要在它上面花调优时间
