# kernel/src

`kernel/kernel.asc` 是提交给判题器的**单文件**，由这里的片段拼出来，不要直接改它。

```bash
cargo asm            # 按 manifest.txt 重写 kernel/kernel.asc
cargo asm --check    # kernel.asc 与源码不一致就非零退出
```

拼接顺序由 [`manifest.txt`](manifest.txt) 决定，**顺序是有意义的**：结果是一个
translation unit，任何声明都必须出现在使用之前。生成文件里每段前面有
`// ===== <path> ====` 横幅，指回它来自哪个源文件。

格式用 Google 风格（根目录 [`.clang-format`](../../.clang-format)，`ColumnLimit: 88`，
`__aicore__`/`__global__` 登记为 attribute macro，`KERNEL_TASK_TYPE_DEFAULT` 登记为
statement macro）。容器里有 clang-format 21：

```bash
find kernel/src -name '*.asc' -exec clang-format -style=file -i {} +
```

每个文件自己开关它用到的 namespace，括号配平——这样 clang-format 能单独处理一个文件，
重排 manifest 也不会把 namespace 拆开。

## 目录

| 目录 | 职责 |
| --- | --- |
| `core/` | 全家共用的词汇：片上预算与 `Plan`/`VTiling`、group list 遍历、反量化原语、退化形状用的标量 helper |
| `host/` | 只在 host 跑的东西：ACL 包装、tiling planner、kernel 标量实参的位打包 |
| `cube/` | cube（AIC+AIV）算法主体。都是 inline 模板，不含 `__global__` kernel |
| `vector/` | AIV-only 的各族。每族自带 kernel 与 launcher，因为它们没有共享主体 |
| `launch/` | cube 的 `__global__` kernel 与 launcher：形状打包、grid、scratch、流同步 |
| `dispatch.asc` | `run_kernel`：按形状选族的分类器 |

## cube 主体与它们的旋钮

四份 cube 主体，各自覆盖一簇由自动搜索调出来的变体；差异做成模板参数而不是复制一份：

| 文件 | 主体 | 谁用它 |
| --- | --- | --- |
| `cube/full_k_body.asc` | `CubeFullKBody` | 短行族（点 2 一类）、128 对齐 tile 族、panel 常驻变体 |
| `cube/unit_body.asc` | `CubeFullKUnitBody` | 窄 N 族、bf16 重打包族及其 tail 变体 |
| `cube/half_body.asc` | `CubeFullKBodyScalarHalf` | fp16 staging 的中等族、fp16 重打包族及其 tail 变体 |
| `cube/half_pair_body.asc` | `CubeFullKUnitBodyScalarHalf` | 256 宽双输出 tile |
| `cube/mix.asc` | `CubeMix` | 兜底：按 slab 过 L1，任何形状都能跑 |

旋钮的含义写在各个主体的注释里。值得单独记住的两个：

* **`UnitFlagChain`** —— 用 unit flag 把 MMAD 直接交给 FIXPIPE（累加中 2，最后一步 3），
  省掉每个 tile 两个 FIX_M/M_FIX 栅栏，代价是整个 kernel 占住 row-major MM 模式。
* **`TightPitch`** —— 反量化按 `Up(nt, 16)` 而不是固定的 BN pitch，让最后一个窄 n-tile
  只搬、只算有效 lane。

## 已知的规则风险

`dispatch.asc` 里有若干门卫按精确的 `(k, n, e)` 三元组选族（1023×1025×7、767×1025×4、
511×513×4、256×512×8），被选中的 kernel 还把这些维度编译期写死
（`launch/cube_packed.asc` 的 `packed_tail`、`launch/cube_full_k.asc` 的 `panel_tiles`）。
这是重构**之前**就有的逻辑，这次原样保留了。按 [`rules.md`](../../rules.md) 2.8
（"逐条精确匹配评测用例编写专属实现"），这部分有被判违规的风险——要不要换成按形状区间
的分类器是策略决定，不是重构能代劳的。
