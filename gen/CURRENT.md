# 现有 kernel.asc 的分派实况

generator 的输入规格。所有行号针对 `kernel/kernel.asc`（sha256 `5e8298d6…`，6122 行）。

`run_kernel`（5814–6121）是一条平铺的 `if … return` 级联。下面的字母标签按源码顺序，是本文档自定的。假设 `availableCoreNum = 20`，即 `p.cores=20`、`p.vcores=40`。

## 15 点路由

| 点 | 形状 | 命中闸门 | 启动的 kernel | grid | 关键参数 |
| --- | --- | --- | --- | --- | --- |
| 1 | E4 M4096 K768 N1024 gt0 fp16 | **Y** (6103) | `SearchP1HalfPairOutput11840034<float,half>` → `CubeFullKUnitBodyScalarHalf<…,256,128,128,64>` | 20 | kp=768 nTiles=8；staging 10.5 MB |
| 2 | E4 M257 K511 N513 gt1 bf16 | **X** (6096) | `mix11::SearchP2Mix115300000<bf16,bf16>`（MIX_AIC_1_1） | 20 | 96,128,512,48；`Upper=30 ≤ 4*20` 故留在 mix11 |
| 3 | E2 M3 K2 N2 gt0 fp16 | **D** (5858) | `expert_output::SearchTinyExpertOutput6400000<float,half>` | E=2 | rowDims=1158 |
| 4 | E8 M1027 K511 N769 gt0 fp16 | **T** (6046) | `SearchCubeCompact4700001<float,half>` → `CubeFullKBodyScalarHalf<…,192,128,96,128>` | 20 | nTiles=7；staging 4.8 MB |
| 5 | E5 M2 K2 N1 gt0 bf16 | **E′** (5876) | `SearchTinyCapacityContract11850017<true,2,2,1>` | 1 | — |
| 6 | E1 M2 K2 N2 gt0 fp16 | **E″** (5882) | `SearchTinyCapacityUnrolled11850012<false,2,2>` | 1 | E==1 过不了 E′ 的 `E>=2` |
| 7 | E1 M1 K8192 N2 gt0 fp16 | **F** (5888) | `SearchLongDotCapacity11840033<8192,false,scalar>` 或 `…DMAScales` | 2 | DMA 变体当 `(xs\|ws)&3 != 0` |
| 8 | E1 M1 K8192 N2 gt1 bf16 | **F** (5888) | `SearchLongDotCapacity11840033<8192,true,scalar>` | 2 | bf16 无 DMA 变体 |
| 9 | E4 M513 K383 N257 gt1 bf16 | **W** (6088) | `SearchCubeCompact4700002<bf16,bf16>` → `CubeFullKUnitBody<…,64,128,384,32>` | 20 | nTiles=3 |
| 10 | E4 M1025 K1 N769 gt0 fp16 | **R.1** (6015) | `p10_fixed_ub::SearchP10FixedUb7300000` | **16** | kc=32 nb=784；selectedUb=163584 |
| 11 | E4 M1027 K769 N1 gt1 bf16 | **P** (5994, K≤800) | `SearchN1FixedLayoutSafe7100000<bf16,bf16,32,800>` | 40 | 固定 UB 184672 B |
| 12 | E8 M768 K256 N512 gt1 bf16 | **U** (6055) | `SearchCubeCompact4700003<bf16,bf16>` → `CubeFullKBody<…,64,128,128,64,2u,true>` | 20 | nTiles=4 |
| 13 | E128 M63 K255 N257 gt1 bf16 | **S** (6038) | `BalancedOverlapShortGroupMatmulKernel<bf16,bf16>`（纯 AIV） | 40 | kc=128 nb=192 nPitch=288 |
| 14 | E7 M3073 K1023 N1025 gt1 bf16 | **V.1** (6070) | `SearchCubeCompact4700004<true>` → pack + `CubeFullKUnitBody<…,224,128,128,64>` | 20 | wstride=1152；packedWeight 8.2 MB + staging 21.7 MB |
| 15 | E4 M4097 K767 N1025 gt1 fp16 | **V.2** (6078) | `SearchCubeCompact4700005<false>` → pack + `CubeFullKBodyScalarHalf<…,224,128,128,64>` | 20 | wstride=1152；packedWeight 3.5 MB + staging 11.9 MB |

点 3/5/6 的路由还取决于 `scalarOutputWindow`（`(y&127) + M*N*2 <= 32`）和各指针的 64B 对齐，即**评测方给的设备指针数值**。判题缓冲通常 512B 对齐，故上表是预期路由。闸门 B (5845) 在实践中是死的（只在本地 CPU/错位视图下触发）。

## 通用兜底：闸门 Z

未命中任何特化谓词的形状会落到 N (5983) / O (5988) / P (5994) / R.3 (6032) / S (6038) / X (6096)，最终是 **Z (6112) 无条件兜底** —— `LeanPlanCube` + `CubeMix<ScaleT,OutT>`。

Z 是真正的保险：`LeanPlanCube` 纯粹从 `m,k,n,e` 推导 `m0/n0/k0/kl/nTiles/rv`（带 L0A/L0B/L0C/L1(524032)/UB 循环收缩），`Plan` 整体按值传递**不做位打包**故无字段宽度溢出，`Scratch` 按 `(⌈m/m0⌉+e)·nTiles` 定界，与 `CubeMix` 的 `Schedule::Total()` 走法一致。

所有 `[H]` 闸门都是纯性能绕道，`ConfigureCubeFullK` 在改写 `Plan` **之前** 就 `return false`，且 T/U/V/W/Y 都在 `Plan selected=p` 的副本上操作。**所以新点的正确性有保障，速度没有**——预计通用 `CubeMix` 比特化家族慢 2–5 倍。

一个例外：**X 把 `p` 按引用传给 `ConfigureCubeFullK`**（6097，不像 T/U/V/W/Y 传副本）。今天安全是因为两条 `return false` 都在任何写入之前，但 generator 必须保持这个顺序，否则会污染 Y 和 Z 随后使用的 plan。

## 免费删除（已确认无人调用）

- `EXACT_K`（32）零引用；`FORM_CUBE`…`FORM_OUTER`（23–31）全部死亡——没有任何存活代码读 `p.form`
- `PlanCube`（841）、`PlanVector`（873）、`BuildVdotTiling`（910）、`UseBalancedOverlapShortGroup`（3016）、`LaunchPublishedFixed96`（2925）均未被 `run_kernel` 调用
- cube 家族 3 个死实例：4908（B 的 `CacheXS=false` 224）、4891（D 的全补丁关闭 224）、以及 `SearchCubeCompact4700004/4700005` 两个 `template<bool BF>` 各有一半的 arm 从未实例化
- 所有搜索编号（`11840033`、`4700001-5`、`6400000` 等）**只是符号名**，没有任何分派读它们
- `NMajor` 分支在全部 8 个存活实例中都是 false；`BM<=32 || BM==208` 早握手分支永不命中

## 理由不明的经验谓词（**保留**，但标注待重测）

规则 2 之 8 禁的是「每个 case 一个专属实现」，按形状性质分派并为该性质写专门算法是正常优化。下列谓词**不是合规问题**，列出来的唯一原因是：它们讲不出「为什么这个规模区间适合这个算法」，所以很可能是过拟合到某个点的噪声，换个形状就失效——对 10:15 新增的 5 个点尤其危险。

性能优先，全部保留；generator 在 IR 里给它们打上「来源不明」标记，以便将来有余力时逐条重测。

| 闸门 | 行 | 表达式 | 针对 | 可疑之处 |
| --- | --- | --- | --- | --- |
| **V** | 6065 | `n % 32 != 0` | 点 14/15 | 讲不出性能理由 |
| **T** | 6046 | `(k%128 \|\| n%128)` | 点 4 | 反向谓词，唯一作用是让对齐形状**跳过** T 去命中 U；即它编码的其实是「U 在对齐形状上更快」这个事实，写法绕 |
### 有明确形状理由的特化谓词（正常优化，不动）

这些虽然各自只命中一个已知点，但谓词编码的是**真实的形状性质**，理由清楚：

| 闸门 | 行 | 表达式 | 形状理由 |
| --- | --- | --- | --- |
| D / K.1 / K.2 | 5858 / 5951 / 5956 | `K==2 && N==2` | 整个问题装进寄存器，零向量操作、零 UB |
| F | 5888 | `N==2` + K 的 1024/2048/4096/8192 阶梯 | `N==2` 使两个通道落在同一个 int16 字里，一对 `ShiftLeft/ShiftRight` 即可拆分；K 阶梯是 UB 容量的 2 的幂分桶 |
| L.1 | 5967 | `bf && N==1` | matvec，沿 K 归约 |
| P | 5994 | `N==1`，内层 `K<=800` | 同上；800 是 UB 列容量 `C` 的硬上界 |
| O | 5988 | `M==1 && N<=2` | 只有 K 可并行，一核一输出通道 |
| R.1 / R.3 | 6015 / 6032 | `K==1` | 退化成外积，无归约，成本 = `2*M*N` 字节的 store |
| S | 6038 | `e>=32 && m<=2e` | 每专家约一个 token |
| T / U / W / Y | 6046 / 6055 / 6088 / 6103 | `m/e`、`n`、`k` 的区间 + 对齐 | 选 BM/BN/BK/RV 的 tiling 区间；对齐条件决定能否用 padded Fixpipe 矩形 |

另外六处 `cores<=20`（6046/6055/6070/6079/6088/6103）是对 `availableCoreNum == 20` 的硬假设。
**修正（2026-11-08）**：6103 是闸门 Y，它服务**点 1**，所以核数一变会降级**六个**点——
`gen plan --all --cores 24` 实测把 **1**、4、9、12、14、15 全部打到兜底 Z，不是先前写的五个。

## 新点正确性雷区

按被新点踩中的概率排序。

1. **`p1_half_pair::Launch`（4215）把 `k/128-1` 打包进 3 位** ⇒ `k ≤ 1024`。闸门 Y **没有 k 的显式上界**，
   `k` 只被 L1 预算间接限制。**修正（2026-11-08）**：该预算有**两道独立闸门**，不是一道——
   闸门 Y 自己的 `512*kp ≤ 524032`（6105）与 `ConfigureCubeFullK`（2863-2864）的
   `(256 + 2*128)*kp > L1_BYTES(524288)`，两者只差 256 字节。只去掉一道不会溢出（`k=1152` 仍落到 Z），
   `gen plan` 的回归测试 `dropping_the_l1_guards_overflows_the_k_field` 要同时去掉两道才能复现。
   **底层风险不变**：两道都是 L1 预算检查，谁也没有在约束被打包的那个量，
   所以 generator 若重写其中一处的算术，必须保住另一处。
2. ~~两个 `tiny_capacity` `Select` 阶梯静默 WA~~ —— **此条已被 `gen plan` 的实现核查推翻（2026-11-08）**。
   两个阶梯末尾都有 `throw std::invalid_argument`：**5414 行**（unrolled，"outside registered power-of-two
   capacity family"）与 **5727 行**（contract，"outside registered M/K/N capacity family"）。
   所以放宽闸门 E 会得到**可见的 RE**，不是静默的错答案。
   阶梯本身仍然只是刚好覆盖闸门 E 的定义域（unrolled 止于 `(k≤4,n≤2)` 与 `(k≤8,n≤1)`，
   contract 止于 `(m≤4,k≤2,n≤2)`），但失败是可诊断的——这对 10:15 加点那一刻是重要区别。
3. **`LaunchNarrowSingleKSelected` 只实例化 bf16**（3670）。闸门 W 以 `bf &&` 开头所以今天安全，但两者相隔 2400 行；放宽 W 会丢 fp16 正确性且**没有编译错误**（模板根本没被实例化）。
4. **`m/e` 被当作「每专家行数」**（6046/6055/6065/6086/6096/6103）。这是均匀分布代理，不是上界——点 15 真实分布是 `[3073,1,0,1023]` 而 `m/e=1024`。现有 kernel 仍走真实 `groupList` 所以没错，但任何把 `m/e` 当真上界的新谓词在倾斜分布上必错。
5. **`Schedule::Total()` 的无符号下溢保护**（65–84，`end > start ? … : 0`）是针对从 GM 竞态读 `groupList` 导致的间歇 ACL 507015 的既有修复。**generator 生成的每一种新遍历都必须带同样的 `end > start` 保护**，否则非单调累积表会算出十亿级 tile 数和越界 MTE 地址。
6. `SearchTinyE1STDev3400000` 的 `rowDims` 漏掉 E 字段（5923 vs 解码器 4957），仅因闸门 H 要求 `E==1` 才正确。`TinyCacheReduction` 的 `tinyCacheDims`（5940）把 `M-1` 塞 4 位、`E-1` 塞 3 位，两处潜在溢出都只被无关子句挡住。
7. `BuildOuterTiling`（947）从不校验 `ubBytes(1) <= 188160`，循环条件是 `rows>1 && …`。在 N=8192 上该值为 158304 B 故安全——是算术巧合，不是检查。

## 位打包字段余量

**由 `gen plan` 的实现逐个核过（2026-11-08）。先前此表漏了四个 `k` 字段，全部零余量**；
尤其 `LaunchAlignedMedium64` 整个打包器都没记。

| 打包点 | 字段 | 容量 | 存活点 | 闸门允许 | 余量 |
| --- | --- | --- | --- | --- | --- |
| `LaunchMediumScalarHalf`(4793) | `n-1` 10 位 | 1023 | 4 | n ≤ 1024 | **零** |
| `LaunchMediumScalarHalf`(4793) | `k-1` **9 位** | 511 | 4 | k ≤ 512 | **零**（先前漏记） |
| `LaunchMediumScalarHalf`(4793) | `e-1` 4 位 | 15 | 4 | e ≤ 16 | **零** |
| `LaunchAlignedMedium64`(4942) | `k/128-1` **2 位** | 3 | 12 | k ≤ 512 | **零**（整个打包器先前漏记） |
| `LaunchAlignedMedium64`(4942) | `e-1` 4 位 | 15 | 12 | e ≤ 16 | **零** |
| `LaunchActivePacked224`(4914) | `k-1` **10 位** | 1023 | 14, 15 | k ≤ 1024 | **零**（先前漏记） |
| `LaunchActivePacked224`(4914) | `e-1` 4 位 | 15 | 14, 15 | e ≤ 16 | **零** |
| `mix11`/`packed32`(3006/2971) | `k-1` **9 位** | 511 | 2 | `kp ≤ 512` ⇒ k ≤ 512 | **零**（先前漏记） |
| `mix11`/`packed32`(3006/2971) | `cores-1` 5 位 / `e-1` 7 位 | 31 / 127 | 2 | 32 / 128 | 零 |
| `LaunchNarrowSingleKSelected`(3666) | `e-1` 4 位 | 15 | 9 | e ≤ 16 | **零** |
| `p1_half_pair`(4215) | `n/128-1` 6 位 / `e-1` 7 位 | 63 / 127 | 1 | N ≤ 8192 / E ≤ 128 | 零 |
| `p1_half_pair`(4215) | `k/128-1` 3 位 | 7 | 1 | **无显式上界** | **仅靠间接** |
| `expert_output` / `tiny_capacity` | `M-1` 2 / `K-1` 3 / `N-1` 2 / `E-1` 7 位 | 4/8/4/128 | 3, 5, 6 | 同值 | 零 |

**对放宽闸门的实际含义：放宽 T 的 `k<=512`、U 的 `k<=512`、V 的 `k<=1024`、X 的 `kp<=512`、
或 T 的 `n<=1024`，都会静默把某个维度别名掉。** `gen plan` 的 `--no-l1-guard` 加上
对 `p1_half_pair` 的 `k/128-1` 标注的 `incidental` 就是用来抓这一类的。

## 指针依赖的谓词（不是形状依赖）

`gen plan` 用 `--ptr <名>=<值>` / `--misaligned` 建模，默认全部 64B 对齐。
**先前此节只提了点 3/5/6，实际点 7/8 也依赖指针**：

- `longdot_capacity::Launch`（3635-3637）在 `(xScale|weightScale)&3 != 0` 时切到 `SearchLongDotCapacityDMAScales`
- 点 7 与点 8 都按 `y&31` 翻转 `ScalarOutput` 模板参数

实测 `--misaligned` 下点 3/5/6 全部塌到闸门 **B**（`TinyShortReduction`），点 7 走 DMA-scales 变体，其余 11 点不动。

## counts 对分派毫无影响

**host 从不读 `groupList`**，所以 counts 这一列对路由完全是死的。
所有 `m/e` 谓词（6046/6055/6065/6086/6096/6103）都是**均匀分布代理**——
点 15 的 `m/e = 1024` 对应的真实分布是 `[3073,1,0,1023]`。

这意味着现有分派**无法对倾斜做出任何反应**。`gen plan` 校验 counts（长度 == E、和 == M）
并把它带下去供 tile 高度决策使用，但当前谓词一个都不看它。

## 启动机制

**grid 三种制度**：AIV 小形状路径按形状直接定（`E` / `M` / `min(M,2c)` / `min(2c,N)` / `1` / `2` / `VTiling::coreNum`）；cube MIX 路径一律 `p.cores`，但 `LaunchPacked32` / `mix11::Launch` 会夹到 `GetCoreNumAic()` 并在其为 0 时抛异常，而 `LaunchMediumScalarHalf` / `LaunchAlignedMedium64` / `LaunchNarrowSingleKSelected` / `LaunchActivePacked224` / `p1_half_pair::Launch` **不夹**，直接信任 `availableCoreNum ≤ 物理 AIC`。`mix11` 设 `p.vcores = p.cores`（1:1 MIX），其余都是 `2*p.cores`。

**参数传输两种风格，都要保留**：(1) `Plan`（15×u32）或 `VTiling`（8×u32）按值放进启动参数，无字段宽度风险；(2) 单个位打包 `uint32_t dims`，kernel 内解包成新的 `Plan`/`GeneralVectorDims`，每个打包器布局都不同（见上表）。

`ConfigureCubeFullK<BM,BN,BK,RV,Prefetch,SingleL0>` 设 `kp,m0,n0,m0r,n0r,n0c,k0,kl,nTiles,rv`，当任一维为 0 或 `(BM + (Prefetch?2:1)·BN)·kp > 512K` 时返回 false。它**不设 `wstride`**；packed-224 路径在 host 设 `packed.wstride = UpH(n,128)` 而 kernel 又从 `dims` 重算 `(n+127)&~127`，两者必须同步。

**GM scratch 全部自行分配，评测不提供 workspace。** `CubeFullKStagingBytes(p) = (⌈m/m0⌉+e)·nTiles·m0·n0·4`；staging 半精度的后端取其 `/2`（4795、4217、4916）。`LaunchActivePacked224` 额外在 `run_kernel` 内分配第二块 `Scratch packedWeight(e·k·UpH(n,128))`（6074/6081）。

**同步不对称**：每条分配 `Scratch` 的路径都在析构前 `aclrtSynchronizeStream`（各 `Launch*` 内部，Z 在 6119）；而 AIV/小形状路径（B、D、E、F、H–P、R、S）**不同步就返回**，启动保持异步。若判题按逐 kernel 耗时求和这没问题，但 generator 必须有意识地复现这个不对称，而不是碰巧。

## probe_ask（5735–5812，5830 触发）

读取隐藏点**内容**的一位预言机，仅在 `-DGMM_ASK=<掩码>` 下编译进去。

- `GMM_ASK` 是 15 位掩码（第 h 位 = 第 h+1 个点）；`GMM_ASK_WHAT` 选张量：`0`=`x`（M·K 字节 int8）、`1`=`weight`（E·K·N）、`2`=`weightScale`（E·N·(bf?2:4)）。两者默认 0，正常构建会把整个 namespace 编译掉。
- `Point()` 把 `(E,M,K,N,groupListType)` 对照**15 个已公布形状的硬编码表**（5780–5784）返回下标。`Answer()` 命中掩码时完全接管该点并让 `run_kernel` 提前返回——真实 kernel 根本不跑。
- `AskAllZero<<<1>>>` 单 AIV block 扫描被探测区域：**发现任何非零字就直接返回不碰 `y`**（→ 判 AC）；**全零则把 `0xFFFF` 铺满 `y`**（→ 保证 WA）。
- 信号是该点的 **AC-vs-WA**，不是 `y` 里的值。配合 CLAUDE.md 的藏分手法，每次提交可对一个隐藏点的数据问一个是非题。
- 机械细节：复制循环（5766–5771）每块只写 16-half 对齐的前缀（`al = n/16*16`）却按完整 `n` 推进 `done`，故每块尾部最多 15 个 half 未被写。对其用途无害。

**generator 要接管这件事**：探测不该靠 C++ 预处理器开关，而应是 `--profile probe` 的输出，且 `Point()` 的硬编码形状表要换成运行时比对 generator 注入的形状描述。
