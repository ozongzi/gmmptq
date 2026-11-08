# kernel.asc generator —— 设计

## 为什么要有它

1. **合规刚需。** 规则 2 之 8 禁止「逐条精确匹配评测用例编写专属实现」。现有 `kernel/kernel.asc` 遍布 `EXACT_K=1024`、精确 M/K/N 等值判断、`SearchP2Fixed964100000` 这类带搜索编号的专属核，赛后人工核验有作废全部成绩的风险。generator 必须按**形状类别**生成通用谓词分派。
2. **新点保险。** 11 月 14 日最后 90 分钟新增 5 个未知点。按点硬编码的谓词不会命中它们；一个真正的形状分类器会。
3. **可维护性。** 四份近似副本（`CubeFullKBody` / `CubeFullKUnitBody` / `CubeFullKUnitBodyScalarHalf` / `CubeFullKBodyScalarHalf`）约 1900 行、占全文 31%，改一次 cube 策略要手工同步四处。
4. **探测代码也归它管。** 探测/自旋变体不该靠 C++ 预处理器开关，而应是 generator 的一个输出 profile。同一套 IR 两个 backend：比赛期变体 + 最终合规提交。

## 占位 sigil

模板占位符用**全角中文括号** `「名字」`。理由：这些码位不可能出现在 C++ / Ascend C 源码里，所以不需要任何转义，也不会和 `{}`、`<>`、`$`、`#` 之类的 C++ 语法冲突。

- `「名字」` —— 标量替换
- `「#名字」` —— 嵌入另一个片段（递归展开）
- `「?条件：甲：乙」` —— 二选一
- 块级标记用 `【名字】`，只在片段清单文件里出现，不进入输出

## 分阶段实施

### 里程碑 0：字节级安全网（先做这个）

`gen emit --profile verbatim` 输出必须与当前 `kernel/kernel.asc` **逐字节相同**（sha256 `5e8298d6a3544cf213d65c3dd8df73a2bc7934be8d5d014ba4710144d79f1bed`）。

做法：把当前 kernel.asc 在顶层边界切成编号片段存进 `gen/chunks/`，加一份顺序清单，`emit` 就是按清单拼接。此时还没有任何参数化——**这一步的全部价值是建立可验证的起点**。

`gen check-chunks` 做这件事的自检，必须进 CI 式的日常检查。

### 里程碑 1..N：逐家族替换

每次只把**一个**家族从 verbatim 片段换成参数化 emitter，然后跑完整验证：

```
编译通过 → 15 点全 AC（真实分组）→ 15 点测时与基线无回退
```

任何一步失败就回退该家族，不往下走。顺序按风险从低到高：先 `preamble`（纯常量）、再 `tiny` 家族、再 `dispatch`、cube 家族**最后**做（它有 27 处手调痕迹）。

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

## profile

| profile | 用途 | 含探测核 | 含自旋核 | 合规 |
| --- | --- | --- | --- | --- |
| `verbatim` | 里程碑 0 的安全网 | 原样 | 原样 | 不保证 |
| `final` | **最终提交** | 否 | 否 | 必须 |
| `probe` | 套隐藏点信息 | 是 | 否 | 否 |
| `spin` | 藏分：其他点自旋到定值 | 否 | 是 | 否 |

`final` 必须**拒绝**生成任何探测/自旋构造，并且拒绝生成按点等值谓词——这是代码层面的合规闸门，不靠人记得删。

## 形状分类器

规则 2 之 8 的约束决定了分类器只能用**与规模相关的通用谓词**，不能用等值匹配。合法的轴：

- `K` 的量级（极小 / 中等 / 极深）与是否 `K == 1`
- `N` 的量级与是否 `N == 1`
- `M` 的量级与是否 `M` 接近 `E`（每专家一个 token）
- 空专家比例（`counts` 中 0 的占比）
- token 分布的不均衡度
- 输出 dtype（fp16 / bf16）
- `groupListType`

非法的（现存代码里要清掉的）：`k == 1024`、`m == 1027`、`n == 513` 这类等值判断，以及任何以搜索编号命名的专属核。

每条谓词都必须能用一句话说明「为什么这个规模区间适合这个算法」。说不出来的谓词就是伪装过的硬编码。

## 不变量（必须全程保持）

- 一次计算只能一个 kernel launch（规则 2 之 12）。若现存 `PackActiveWeightRowsInline` 是独立 launch 后再接矩阵乘，必须合并。
- Host 侧只做参数校验、shape 推导、tiling、启动配置；不得读输入张量、不得生成预查表（规则 1、2 之 1/3）。
- 不得依赖未初始化内存或旧输出（规则 2 之 7）。
- 不得按数据规律跳算（规则 2 之 5）——包括运行时检测 x 全 0 后跳过矩阵乘。
- cube 家族的 27 处手调痕迹（见 agent 报告的 R1–R27）每一处都要有对应的保活测试或 golden 比对。
- UB 槽位表由 generator 负责：必须生成「每个槽位都在 `UB_USED` 内且互不重叠」的编译期断言，因为 `Static=true` 的零成本 pipe 把偏移责任完全交给了 generator。
