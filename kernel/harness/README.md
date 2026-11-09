# kernel/harness

容器里那套测量设施的**备份**。原件在 devenv 容器 `~/gmm-rebuild/harness/`（`abba.sh`、`verify.sh`
原本在上一级 `~/gmm-rebuild/`），不进 git 就只存在于一台随时会没的容器里。这里逐字节相同，
用的时候拷回去，别在 Windows 上改完忘了同步。

`judge_shim.h` 是**从判题器反推出来的**，判题器不发这两个 struct，重推一次很贵——这是备份这个
目录的主要理由。

## 每个文件干什么

| 文件 | 作用 |
| --- | --- |
| `judge_shim.h` | 判题器交给 `run_kernel` 的 `TensorInfo` / `TensorGroupInfo`，以及 dtype 编码（0=fp32 1=fp16 2=bf16 3=int8 6=int64）。从提交自己的 `Tensor()` 校验反推。 |
| `driver.asc` | 本地驱动：读一个 case 目录，`#include "kernel.asc"` 把提交原文编进来，在真卡上跑 `--iters` 次，把 `y` 写回 `out.bin`。**不在这里计时**。输出缓冲先用 `0xAB` 涂毒，漏写的尾巴会留证据而不是蒙到一个像样的 0。 |
| `CMakeLists.txt` | 编 `driver`，`find_package(ASC)` + `--npu-arch=dav-2201`。 |
| `profile.sh` | 唯一的计时入口：`msprof` 取每次 launch 的 device 时间，报 min / median / max / spread 和 cube、vec、blocks 计数器。 |
| `sweep.sh` | 正确性：把一个 driver 跑遍一棵 case 树，逐点 PASS/FAIL + 最大相对/绝对误差。deadlock 由 `timeout` 兜住（`GMM_TIMEOUT`，默认 120 s）。 |
| `compare.sh` | 15 个点各跑一遍 `profile.sh`，和 tbest / floor 并排打一张表。 |
| `ab.sh` | 一个 sed 改动的 A/B：在变体目录里就地改 `kernel.asc`、重编、先验正确性再报时间，跑完 `base.asc` 还原。 |
| `examples.sh` | 跑 problem.md §6 的三个手算例子（`ex_two-experts`、`ex_empty-head-mid-tail`、`ex_negative-and-zero`）。这是唯一不由我们自己代码产生的 ground truth。 |
| `verify.sh` | 一行一个变体的正确性总览：真实分组 15 点 + 合成 75 例，各报一行 `pass/fail/crash`。 |
| `abba.sh` | ABBA 交错对比两个变体的 median，逐点逐 rep 写 TSV。 |

为什么时间不从 host 取：这些核就跑一两微秒到几十微秒，一次 launch + synchronize 的 ACL 开销就是
几十微秒，host 墙钟测的是 ACL 不是 kernel。`profile.sh` 这条路复现过官方调优指南的 218.1 µs，
误差 3% 以内——这是我们唯一的标定。

## 先 source 环境

每个新 shell 都要，否则 `cmake` 找不到 `ASC` 语言、`msprof` 不在 PATH：

```bash
source $HOME/Ascend/cann-9.0.0/set_env.sh
```

`abba.sh` 和 `verify.sh` 自己会 source（并且自己 `cd ~/gmm-rebuild`），其余脚本不会。

## 建一个变体目录

一个变体 = 一份 harness + 一份 `kernel.asc` + 自己的 `build/`。目录名约定 `h-<标签>`：

```bash
cd ~/gmm-rebuild
mkdir -p h-foo && cp harness/* h-foo/ && cp foo.asc h-foo/kernel.asc
cmake -S h-foo -B h-foo/build && make -C h-foo/build -j8
# 产物：~/gmm-rebuild/h-foo/build/driver
```

改完 `kernel.asc` 只要 `make -C h-foo/build -j8`。`ab.sh` 必须放在变体目录里跑（它按
`dirname $0` 定位 `kernel.asc` 和 `build/`），`cp harness/*` 已经把它带过去了。

## 命令行

```bash
cd ~/gmm-rebuild
source $HOME/Ascend/cann-9.0.0/set_env.sh
C=cases-b40da251a039                 # 真实分组那棵树，见下面的坑
D=$PWD/h-foo/build/driver

# 测一个点（20 次 launch，min 最干净，median 最老实）
bash harness/profile.sh $C/tp14_even $D 20
bash harness/profile.sh $C/tp14_even $D 20 24      # 只给 24 个 core

# 15 个点全测一遍，和 tbest 并排
bash harness/compare.sh $C $D 20 even

# 正确性：一棵树全扫（iters=1 够了）
bash harness/sweep.sh $C $D $PWD/gmmbench 1
bash harness/sweep.sh $PWD/even $D $PWD/gmmbench 1      # 合成 75 例
bash harness/sweep.sh $C $D $PWD/gmmbench 1 4 9 13       # 只扫 4/9/13
bash harness/verify.sh foo bar                           # 多个变体的总览

# §6 手算例子
bash harness/examples.sh $C $D $PWD/gmmbench

# 一个 sed 改动的 A/B（在变体目录里）
cd h-foo && ./ab.sh ~/gmm-rebuild/$C ~/gmm-rebuild/gmmbench "bn256" 's/BN = 128/BN = 256/' 1 4 14 15

# ABBA 交错对比两个变体：2 轮，每次 20 samples
bash abba.sh 2 20 cmp.tsv 1 4 14 15
```

`gmmbench` 是 [`fixture`](../../fixture/README.md) 那个零依赖二进制，scp 进容器即可；case 树也由它生成。

## 坑

**1. `compare.sh` 里的 tbest / floor 数组是旧的。** 写死在脚本里的 `TBEST=(...)` 来自早期的
shape.md，和 [`UPDATES.md`](../../UPDATES.md) 已经差很远（点 1：31.06 vs 27.921；点 9：16.91 vs
**0.464**，差 36 倍；点 14：64.79 vs 52.402）。那一列只能当两个 kernel 互相比较时的刻度，
**不能当成"离最优还差多少"**。要看真实差距去读 `UPDATES.md`。另外 tbest 测在判题机上、
这里测在容器上，两台机器可不可比**从未验证过**，所以那个比值只是刻度，不是判决。

**2. 两棵 case 树，`_even` 后缀会骗人。**

- `cases-b40da251a039/tp*_even` —— **里面是真实分组**。目录后缀和 `meta.txt` 是旧生成器留下的
  标签，没跟着数据更新：`tp04_even/meta.txt` 写 `counts [129,129,...]`，而 `grouplist.bin` 实际是
  `897 898 915 915 980 1011 1011 1027`，即真实分组。认 `manifest.json` 的 `observed_sha256`
  （= 目录名那串 `b40da251a039…`），别认 `meta.txt`。
- `even/`（75 个目录，15 点 × `even/skew/empty/tail/head`）—— **真的是合成压力形状，不能用来计时**。
  只拿来查会把 kernel 搞挂的情况。拿它调优会得到反向结论：2026-11-07 点 9 的一个候选在 `skew`
  上快 17.9%，在真实分组上慢 4.18%。

所以：计时只用 `cases-b40da251a039`，`even/` 只过 `sweep.sh`。`fixture` 新生成的树目录名是
`tp*_real`，和脚本里写死的 `_even` 对不上——`compare.sh` 的 split 可以从第 4 个参数传
（`compare.sh ~/gmmcases $D 20 real`），`ab.sh` 和 `abba.sh` 里是硬编码的，得改。

**3. `abba.sh` 全是写死的。** `o1=prod; o2=k9` 两个变体（即 `h-prod` 和 `h-k9`）、`_even` 后缀、
case 根 `cases-b40da251a039`、`cd ~/gmm-rebuild`。换一对变体就得改脚本，它不收这些参数。

**4. `profile.sh` 的 profiler 配置不是判题器那一套。** 它开的是
`--aic-metrics=ArithmeticUtilization --ai-core=on`；和判题器 50 个字段对齐过的配置在
[`../cal/cal.py`](../cal/cal.py) 的 `PROFILER_CLI` 里。两者的绝对值不能混着比。

**5. `profile.sh` 一个进程里 launch `iters` 次，量的是 warm 状态。** 判题器是冷的。
`cal.py` 的 `fresh` / `hot` / `cold` 三档就是为了量这个差，排名候选只认 `cold`。
