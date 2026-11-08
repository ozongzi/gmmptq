# fixture

本地测试数据 + 参考答案。**零依赖 Rust**，交叉编译进 devenv 容器。

```bash
cargo fixture cases                  # 15 个点的形状和真实分组
cargo fixture suite ~/gmmcases real  # 生成输入 + golden + 清单（1.2 秒，116 MB）
cargo fixture check ~/gmmcases       # 和清单比，证明这棵树就是对过账的那一份
```

零依赖是刻意的：容器没有 cargo、上不了 crates.io，而且**不需要项目 checkout、不碰数据库、不联网**，
scp 一个二进制过去就能跑。链接器用 `zig cc` 顶，见 `lab/README.md` 的交叉编译一节。

## 分组：只有 `real` 能用来计时

`groups.rs` 里是**判题器自己的分组**，15 个点全部恢复过，点 14 由一次实际执行逐元素确认。
来源是 `data/recovered.json`，`SOURCE_SHA256` 钉住快照；数字写在 Rust 里而不是运行时解 JSON，
这样 `counts_cover_m` 和 `encoding_matches_the_recorded_raw` 在测试时就能查出错。
只有分组是判题器的，x / weight / 两个 scale 仍是我们自己生成的——判题器从不返回这些。

另外五种（`even` `skew` `empty` `tail` `head`）是**正确性压力形状**，覆盖会把 kernel 搞挂的
情况（全挤在一个专家、首/中/尾有空洞），不代表判题器跑什么。**拿它们调优会得到反向结论**：
2026-11-07 点 9 的一个候选在 `skew` 上快 17.9%，在真实分组上慢 4.18%。

`group_list` 按每个点自己的 `groupListType` 编码（0 是累计结束位置，1 是各专家 token 数），
所以编码只有一处实现，不可能和分组对不上。

## 数据不进 git，清单进

生成是确定性的：RNG 种子取自「这条流是干什么的」而不是计数器，所以新增一个 tensor 不会挪动
已有数据的字节，换一台机器跑出来的也是同一批字节。

所以**数据从不保存、从不传输**，每个容器现场生成；进 git 的是 `manifest.txt`（几 KB，
一行一个文件的 sha256）。流程是：在一台机器上生成、和参考实现对过账、`manifest` 存进 git；
之后任何容器 `suite` 完 `check` 一遍，就知道这棵树和当初对过账的那一份逐字节相同。

`check` 不看 `out.bin`、`driver.err`、`*.tmp` 这些跑出来的东西，只认 fixture 本身。

这一套是从 `bench/` 的 gmmbench 加 `search/actual/prepare.py` 整理来的。核对过：15 个点的
8 个数据文件（`x/weight/xscale/wscale/grouplist/golden/golden_f32/shape`）与旧流水线
**逐字节相同**，共 120 个文件；只有人类可读的 `meta.txt` 因为 `split` 那一行写 `real` 而不同。

## 精度判据两种读法都报

题面说「相对误差 < 1e-3，绝对误差 < 1e-3」。字面读是两者都要满足，但一个 50000 的值要绝对
误差 < 1e-3 比类型本身的粒度还细，不可能是原意。通行读法是任一满足即过。`verify` **两个都报**，
不替你选：`失败(宽松读法)` 和 `失败(严格读法)`。

golden 必须自己做舍入：bf16 的表示误差就是 2^-8 = 3.9e-3，已经超了 1e-3，所以不能拿 f32 的
精确值去比，要先按输出类型做 round-half-to-even 再比。`num.rs` 里 f32↔f16 是手写的
（次正规数 + RNE + 溢出到 inf），测试覆盖全部 65536 个 f16 位模式的往返。

## scale 上限是算出来的

题面同时承诺两件事：scale 在 [1e-5, 1]，且每个反量化结果 |v| ≤ 60000。这两条不独立——
全范围随机 int8 的累加器标准差是 `5461*sqrt(K)`，K=8192 时到五十万，scale 接近 1 会直接
冲破 60000。所以按 K 逐例压上限、保证 8σ 仍在承诺内，而且 `golden` 会**断言**这个承诺成立：
生成器要是错了，后面所有测量都是对着判题器永远不会给的数据做的。

## 命令

| | |
| --- | --- |
| `cases` | 15 个点的形状、gt、输出类型、真实分组 |
| `gen <dir> [case] [split]` | 写五个输入 tensor |
| `golden <dir> [case] [split]` | 算参考结果（多线程，精确） |
| `verify <dir> <out.bin> <case> <split>` | 比对，两种读法都报 |
| `dump <dir> <out.bin> <case> <split> [n]` | 逐元素并排看 |
| `suite <dir> [split]` | gen + golden + 自校验 + manifest |
| `manifest <dir> [case] [split]` | 写 `manifest.txt` |
| `check <dir> [case] [split]` | 和 `manifest.txt` 比 |
| `examples <dir>` | 题面三个手算例子 |
| `exverify <dir> <name> <out.bin>` | 和题面的答案比 |
| `custom <dir> <name> E M K N gt float16\|bfloat16 [split]` | 任意形状，二分哪一维挂 |
| `dverify <dir> <out.bin>` | 形状从目录读，不用再传参数 |

`case` 省略是全部 15 个点，`split` 省略是全部 6 种分组。

## 题面自带的三个例子

`examples` 写出的是题面 §6 的三个手算例子，**唯一不是我们自己代码产生的真值**：

| name | 覆盖什么 |
| --- | --- |
| `two-experts` | 两个专家、per-token 和 per-channel scale 都不同，fp16 |
| `empty-head-mid-tail` | 专家 0/2/4 同时为空（首、中、尾），**bf16 的 weightScale** |
| `negative-and-zero` | 负结果和全零输入行都必须保留 |

## 还没做的事

官方调试包里的 `reference.py` 是**官方参考实现**，收 numpy 数组直接调
`reference.impl(x, weight, xScale, weightScale, groupList, groupListType=.., outputDtype=..)`，
不需要它那个 CSV。拿它和这里的 golden 对一次账，就能把「我们的 golden 没系统性错误」从自证
变成官方背书——对完存哈希，之后容器里只用 `check`，torch 不用进容器。**这一步还没做。**
