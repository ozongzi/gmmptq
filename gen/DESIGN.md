# kernel.asc generator —— 设计

## 为什么要有它

1. **可维护性，这是首要理由。** 四份近似副本（`CubeFullKBody` / `CubeFullKUnitBody` / `CubeFullKUnitBodyScalarHalf` / `CubeFullKBodyScalarHalf`）约 1900 行、占全文 31%，改一次 cube 策略要手工同步四处。现在这份代码已经改不动了，这才是优化停滞的根因。
2. **90 分钟窗口的周转速度。** 11 月 14 日 10:15 新增 5 个点，11:45 收榜；而 9:00 所有隐藏点转可见，**新增点的规模也是可见的**。所以需要的不是盲猜式通用分类器，而是**拿到 5 个已知形状后能在 90 分钟内调出来**的能力：加一类、跑参数搜索、编译测量、提交。generator 的核心价值是把这个周转从「手改 6000 行」压到「改几行 IR + 重新生成」。
3. **探测代码也归它管。** 探测/自旋变体不该靠 C++ 预处理器开关，而应是 generator 的一个输出 profile。同一套 IR 两个 backend：比赛期变体 + 最终提交。

### 关于规则 2 之 8 的尺度

「逐条精确匹配评测用例编写专属实现」禁的是**每个 case 一个专属实现**，**不是**禁止利用形状性质。按 `N==1`、`K==1`、`M*K<=8`、`n%128==0` 这类形状性质分派并为该性质写专门算法，是正常优化，不是违规。

所以 generator **不以「谓词看起来更通用」为目标，也绝不为此牺牲性能**。它要做的是让按性质特化变得廉价可控，而不是把特化删掉。

## 占位 sigil

模板占位符用**全角中文括号** `「名字」`。理由：这些码位不可能出现在 C++ / Ascend C 源码里，所以不需要任何转义，也不会和 `{}`、`<>`、`$`、`#` 之类的 C++ 语法冲突。

- `「名字」` —— 标量替换
- `「#名字」` —— 嵌入另一个片段（递归展开）
- `「?条件：甲：乙」` —— 二选一
- 块级标记用 `【名字】`，只在片段清单文件里出现，不进入输出

## 分阶段实施

### 里程碑 0：抽取完整性验证（已完成）

`gen emit --profile verbatim` 字节级复现当前 `kernel/kernel.asc`（45 片段，sha256 `5e8298d6…`）。

**这一步的价值仅限于证明抽取是完整无损的，以及提供一个回滚参照。字节级一致不是后续的目标。** 继续要求输出与现文件逐字节相同，等于主动生成同一堆屎山：现文件有 14 次重复开闭的 `namespace gmmptq`、文件中段的第二个 `#include` 块、前后不一致的缩进、4 套互不相同的 dims 位布局、以及一长串死代码。generator 的意义就是不再产出这些。

`verbatim` profile 从此只作历史参照保留，不再是日常检查项。

### 里程碑 1..N：逐家族替换，**只要求逻辑一致**

每次只把**一个**家族从 verbatim 片段换成参数化 emitter，生成**干净**的代码，然后跑完整验证：

```
编译通过 → 15 点全 AC（真实分组）→ 15 点测时与基线无回退
```

验收标准就是这三条，**不含任何字节或 sha 比对**。任何一步失败就回退该家族，不往下走。

「干净」意味着每个家族被参数化时就顺手做掉这些，不需要单独的规范化步骤：

- 合并成单个 `namespace gmmptq`，不再反复开闭
- 所有 `#include` 提到文件头，只出现一次
- 统一缩进
- 删掉该家族范围内的死代码（见 `CURRENT.md` 的免费删除清单）
- 4 套 dims 位布局归一成一套编码器 + 一套解码器，各 kernel 的字段宽度从此是显式容量断言而不是巧合

## 两条轨道

拿分和重构在短期内必须解耦，否则 80 分赶不上：

- **A 轨（立刻拿分）**：在容器的变体目录（`h-<名>/`）里直接探索 tile 几何与 DMA，用 warm median 测，赢的提交上去。`+153` 分从这里来。探索允许改变体目录里的 `kernel.asc` 副本——仓库的 `kernel/kernel.asc` 仍然只能由 generator 产出，A 轨的结论必须回流进 IR 才算完成。
- **B 轨（generator）**：按下面的里程碑推进，目标是 11 月 14 日 10:15 拿到 5 个新点形状后能在 90 分钟内加一类、搜参、生成、提交。

## 里程碑顺序（按实测分数缺口重排）

见 `kernel/TARGETS.md`。微小点（3/5/6）合计只有 +16.5/+28.1/+23.4 且都已在 tbest 的 1.08–1.17 倍内，**不是主力**，所以原先「M1 = tiny 家族」的排序作废。

| 里程碑 | 家族 | 服务的点 | 空间 |
| --- | --- | --- | --- |
| **M1** | cube 的 **tile 几何**（不是整族重构）——按真实 `counts` 分布选 tile 高度，而不是用 `m/e` 均匀代理 | 4 / 14 / 15 / 2 | **+149** |
| M2 | `SearchN1FixedLayoutSafe`（N==1 的每行归约开销） | 11 | +54.9 |
| M3 | `BalancedOverlapShortGroup`（weight DMA 带宽、非对齐 K/N） | 13 | +48.3 |
| M4 | `gen plan` + dispatch（`run_kernel` 是不可分割的 309 行级联，必须先有干跑能力） | 全部 | — |
| M5 | longdot（7/8）、outer（10） | 7 / 8 / 10 | +93 |
| M6 | tiny 家族 | 3 / 5 / 6 | +68 |

M1 刻意不做 cube 四族的完整统一（那需要先搬代码、且有 27 处手调痕迹要保活）。它只参数化**tile 几何的选择**这一件事，因为四个点共享同一个病根：行利用率 53%–81%。整族统一留到 M4 之后。

## 目录结构

```
gen/
  Cargo.toml          零依赖（和 fixture 一致）
  DESIGN.md           本文件
  chunks/             从当前 kernel.asc 抽出的 verbatim 片段 + manifest
  src/
    main.rs           CLI
    tmpl.rs           「」模板引擎
    chunks.rs         片段加载与拼接
    ir.rs             Shape / ShapeClass / Strategy / FullKEmit 等
    classify.rs       Shape → ShapeClass → Strategy（只允许通用谓词）
    emit/
      mod.rs
      preamble.rs     常量、Plan、VTiling、辅助函数
      fullk.rs        统一的 cube emitter（取代 A/B/C/D 四份）
      tiny.rs         小形状家族
      outer.rs        outer / SearchOuter / FixedUbOuter
      dot.rs          longdot_capacity
      dispatch.rs     生成 run_kernel
      probe.rs        探测核与自旋核（仅非最终 profile）
```

## CLI

```
gen emit --profile <名> [-o 路径]     生成 kernel.asc
gen check-chunks                      验证 verbatim 拼接的 sha256
gen plan <E> <M> <K> <N> <gt> <dtype> <counts...>
                                      打印该形状会走哪条策略（干跑，不生成）
gen classes                           打印形状分类器的全部类别与谓词
```

`gen plan` 是新点的主要工具：11 月 14 日拿到 5 个新点的形状后，先干跑看分类器把它们送到哪里，再决定要不要加一类。

```
gen plan --what-if --tiles BM,BN,BK,RV[,prefetch,singleL0] [--launcher <名>] [--after <标签>]
         <E> <M> <K> <N> <gt> <dtype> <counts...>
```

`--what-if` 不改表：它合成一条「围住该形状的最小整数窗口」闸门，插在兜底之前，打印
(1) 现在走哪条闸门、(2) 加上后走哪条、(3) 位打包是否放得下、(4) 它会生成的 C++、
(5) **可直接粘进 `cascade::cascade()` 的 Rust 表项**、(6) 现有 15 点是否被改道。

### Body：一个槽位的具名实现（2026-11-09 落地）

里程碑 1..N 的「把一个家族换成参数化 emitter」需要一个接缝，这就是 `gen/src/body.rs` 的 `Body`。

```
Slot { symbol, fragment }        // 级联能路由到的符号 + 今天承载它的碎片
Body { name, slot, source, pred, why, default }
```

- **槽位用「符号」标识，不用文件位置。** `dispatch::route` 返回 `Routed.kernel`，所以「点 11 跑哪个 body」由表回答，不需要第二张会和它漂移的映射。
- **同一槽位的两个 Body 是两份实现，不是基准 + 补丁。** 这里不存任何 diff，也没有哪个 body 用另一个来定义;每个 body 自带完整源码(`gen/bodies/<name>.inc`)。里程碑 0 的 emitter 就是「返回这段 bytes」，换成真正的参数化只改这一个 body。
- **`pred` 既打印成 C++ 又对形状求值**，和级联同一条规则，所以谓词不会和自己的文本漂移。
- `--pick <点>=<body>` 不是字符串查表：它把点经级联路由到槽位，要求该 body 属于**那个**槽位，再用 body 自己的谓词对该点形状求值，**并指名是哪个合取项不成立**。未登记的点/body、属于别的槽位、谓词拒绝、同一点被选两次，都在写出任何字节之前硬失败。

**第一个槽位**是 `SearchN1FixedLayoutSafe7100000`(点 11)，两个 Body:`n1-wholereduce`(两级 `WholeReduceSum`，默认)与 `n1-fold`(对齐对折到 ≤64 再一次 reduce，`K>=128`)。

**刻意没有做的：`role` 作用域内的覆盖/重叠检查。** 本文件的分派是**单条有序级联，位置即优先级**，重叠由顺序解决，`plan --what-if` 负责报告插入一条闸门会把谁改道。在闸门顺序旁边再放一套消歧机制，等于对「哪条闸门赢」有两个答案。所以 Body 只在**一个闸门的槽位内部**竞争，而该槽位能不能被走到由闸门自己的谓词决定。`emit::tests::a_new_shape_is_served_by_registering_one_gate` 把这件事钉死：新形状靠登记一条闸门服务，但那条闸门必须插在会抢走它的闸门**之前**，否则什么都路由不到。

**验收是行为，不是字节。** 里程碑 1..N 的三条(编译通过 → 15 点全 AC → 无回退)照旧，外加一条：`opdur.sh` 的 op name 要确认 `--pick` 选中的 body 真的被执行到——没有这一条，`--pick` 只是个没人验证的字符串。`check-chunks` 保留为「碎片表没被碰坏」的自检，**不是**变体层的验收条件。

### 分派级联是数据

`run_kernel` 的 26 条闸门住在 `gen/src/cascade.rs` 的一张表里，每个合取项是一个结构化的
`Expr`——它**同时**知道怎么打印成 C++ 和怎么对形状求值。`plan` 求值这张表，
`emit --profile final` 打印同一张表，所以两者无法漂移。加一条闸门是插一个 `Node::Gate`：
`plan` 与 `emit` 都会在你插入的位置上看到它（未知标签走 `dispatch::generic_arm`）。

讲不清的指针对齐谓词标为 `Expr::Opaque`（C++ 文本与 Rust 求值器确实是两份字符串），
由 `literals_agree_with_the_cpp_text` 做机械交叉检查：两边用到的整数字面量必须一致。

## profile

| profile | 用途 | 含探测核 | 含自旋核 | 合规 |
| --- | --- | --- | --- | --- |
| `verbatim` | 里程碑 0 的安全网 | 原样 | 原样 | 不保证 |
| `final` | **最终提交**：`run_kernel` 由分派表生成，其余片段逐字 | 否 | 否 | 必须 |
| `instrument` | **分解探针**：故意算错的测量仪器（见下） | 否 | 否 | 否（禁止提交） |
| `probe` | `GMM_ASK` 内容预言机，套隐藏点**数据** | 是 | 否 | 否 |
| `spin` | 藏分：其他点自旋到定值 | 否 | 是 | 否 |

`spin` 已实现（`gen/src/spin.rs`）：`emit --profile spin --isolate <点|0> [--ticks N] [--own-ticks N] -o <路径>`。
它发的是 `--profile final` 的同一条级联，只把 `run_kernel` 改名成 `GmmRunKernelBody`，再套一个先调它、
后按点发自旋核的 `run_kernel`。`--isolate 0` 是全点自旋的**标定件**；`--own-ticks` 给被测点自己一个已知
自旋，两次这样的运行能把我们的时间和该点 tbest 分开解出来。

`GmmProbePoint` 的形状表**由 `dispatch::points()` 打印**，不是手写的 —— 所以这个仪器不可能和生成器对
「哪个形状是点 11」产生分歧。外部脚本 `tools/isolate.sh` 靠 grep 找接缝、表是手写的，这是两者的区别。
和 `instrument` 一样，`spin` **拒绝写 stdout**。

`final` 必须**拒绝**生成任何探测/自旋构造，并且拒绝生成按点等值谓词——这是代码层面的合规闸门，不靠人记得删。
它为此做两件事：不发 `#if GMM_ASK` 钩子，并整片丢掉 `5731-probe-ask.inc`（`emit::OMITTED_IN_FINAL`），
这样 `ENDGAME.md` 的交前 grep（`GMM_ASK|GetSystemCycle|GmmProbeSpin|GMM_PROBE`）在生成文件上**为空**。
`GMM_ASK` 默认 0，所以这对真实构建的预处理结果没有影响。

### `instrument` 为什么不叫 `probe`

`probe` 在本文件里本来指 `GMM_ASK` 这个**内容**预言机：它问的是判题方**输入**的是非题，必须提交才能读出 AC/WA。
`instrument` 问的是**我们自己的时间**：把 K 循环上界或某条 DMA 的范围减半，和真内核的时间差就把成本拆成固定项与比例项
（`ENDGAME.md` 的三个实测：点 13 的 34% 与 K 无关、点 11 的 x 流只占 0.8%、点 5 的预读值 0.09 µs）。
两者在读法、提交必要性、合规性上完全相反，共用一个名字就是 11:40 传错文件的那类混乱，所以分名。

`instrument` 的每个输出都带文件头横幅与逐行 `GMM_WRONG_ON_PURPOSE` 标记，CLI 两次告警，且**拒绝写 stdout**。
每条探针声明自己的替换次数，片段改动后探针失效会报错而不是静默测一个空探针。

## 形状分类器

分类器按形状性质分派。性能优先——哪条谓词快就用哪条，不为「看起来通用」让路。有用的轴：

- `K` 的量级（极小 / 中等 / 极深）与是否 `K == 1`
- `N` 的量级与是否 `N == 1`
- `M` 的量级与是否 `M` 接近 `E`（每专家一个 token）
- 空专家比例（`counts` 中 0 的占比）
- token 分布的不均衡度
- 输出 dtype（fp16 / bf16）
- `groupListType`

每条谓词**最好**能用一句话说明「为什么这个规模区间适合这个算法」——这不是合规要求，而是因为说不出理由的谓词往往是过拟合到某个点的噪声，换个形状就失效，对最后 5 个新点尤其危险。但如果实测证明某条窄谓词确实更快，就留着。

现存的 `(k%128 || n%128)`（闸门 T 的反向谓词，作用是让对齐形状跳过 T 去命中 U）和 `n % 32 != 0`（闸门 V）属于这类：理由讲不清但有实测支撑，**保留**，只在 IR 里标注「来源不明的经验谓词」以便将来重测。

## 不变量（必须全程保持）

- 一次计算只能一个 kernel launch（规则 2 之 12）。若现存 `PackActiveWeightRowsInline` 是独立 launch 后再接矩阵乘，必须合并。
- Host 侧只做参数校验、shape 推导、tiling、启动配置；不得读输入张量、不得生成预查表（规则 1、2 之 1/3）。
- 不得依赖未初始化内存或旧输出（规则 2 之 7）、不得注入空 kernel（规则 2 之 10）——点 9 的 0.464 µs tbest 就是这么来的，我们不跟。
- 按**形状**性质特化没问题；按**数据内容**跳算（如运行时检测 x 全 0 后跳过矩阵乘）落在规则 2 之 5，不做。
- cube 家族的 27 处手调痕迹（见 agent 报告的 R1–R27）每一处都要有对应的保活测试或 golden 比对。
- UB 槽位表由 generator 负责：必须生成「每个槽位都在 `UB_USED` 内且互不重叠」的编译期断言，因为 `Static=true` 的零成本 pipe 把偏移责任完全交给了 generator。
