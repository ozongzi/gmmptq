# lab

判题器代理。一个二进制 `lab`，一份 git 同步的文本日志，一个可以随时删掉的 SQLite 缓存。

```bash
cargo lab --help          # 仓库根目录或任意子目录；别名在 .cargo/config.toml
cargo lab quota -a ozongzi
```

仓库根是一个 cargo workspace，产物在根目录的 `target/release/`。另一个成员是
[`fixture`](../fixture/README.md)——本地数据和参考答案，零依赖、能进容器，和 `lab` 互不牵连。

Windows、Linux、macOS 同一份代码，没有平台相关的调用。

## 东西放在哪

**只有一条规则，没有任何环境变量或参数能改它**：项目根目录是可执行文件往上第一个含
`secrets/accounts/` 的目录。`cargo lab` / `cargo build` 的产物在根目录 `target/release/` 下，所以总能
找到自己所在的 checkout。找不到就报错退出，**不会在别处悄悄新建一个空库**——旧版默认
写 `~/gmmlab/gmmlab.db`，仓库搬家时没跟着走，整个历史就这么丢了。

```text
lab/data/log/<机器名>/<UTC 日期>.jsonl   真相：一行一条记录，只有这台机器往里追加
lab/data/body/<sha>.json                响应体，按内容寻址
lab/data/kernel/<sha>.asc               送出去的 kernel 原文，按内容寻址
lab/gmmlab.db                           缓存，gitignore；删掉下次打开自动重建
secrets/accounts/<账号>.json             凭证，进 git
```

## 为什么是这个形状

**日志就是接口。** 对上游的每一次 HTTP 往返——提交、轮询、拉排行榜、拉提交列表、
OAuth 刷新——在发生的那一刻整条写进日志：请求体、状态码、响应头、响应体逐字节原样。
失败也写，401 也写，会话过期时那张 HTML 登录页也写。`lab/data/` 里的东西只追加，
从不改写、从不删除。

**多台机器同步不会冲突。** 每台机器只往 `log/<自己的机器名>/` 追加，其余文件名就是
内容的 sha，所以 Mac、Windows、服务器各自 `git pull` 永远是快进或干净的 rebase。
SQLite 是二进制，放进 git 一定冲突，所以它只是缓存：每次打开把没读过的日志行读进来
（只读到最后一个换行，正在写的半行留给下一次），表结构一变就整个扔掉重建，不做迁移。
`lab/data/.gitattributes` 关掉了换行转换——kernel 原文按 sha 命名，CRLF 一转字节就不对了。

**所有派生量都是读时的视图**：每个用例的 time / best / score、visible/hidden 拆分、
排行榜每一行、每个用例见过的历史最优，全是 JSON1 打在存下来的响应体上。这份历史已经
被重新解释过两次（噪声模型变了），重新查询不要钱，重新提交要配额。

**打分公式只存在一处。** `score.rs` 的 `case_score` / `ratio_from_score` 同时注册成
SQLite 的 `gmm_score` / `gmm_ratio`，视图和 Rust 算的是同一个数。

**stdout 不是接口。** 每个子命令都有 `--json`，`lab sql` 是只读出口，直接吐 JSON。

## 账号

需要联网的命令都必须写 `-a/--account <名字>`，对应 `secrets/accounts/<名字>.json`，
**没有默认账号**。写错了报错里会列出现有的账号。每条提交记下账号名和当时那个账号在
判题器上的 uid。只读本地的命令（`status`、`history`、`source`、`schema`、`sql`）
不需要账号；`status --refresh` 用这条提交当初的账号去查。

## 配额和冷却

每天 50 次（UTC+8 午夜翻页），两次之间 180 s。**以判题器为准**：提交前先拉一次这个
账号在判题器上的提交列表，再加上本机已经占了、判题器还没列出来的名额（刚占的、POST
失败的）。几台机器共用一个账号时，本地日志在 pull 之前永远是不全的，只看本地会偏乐观。
本机两个进程同时提交时，占名额是同一个 SQLite 事务，不会都以为自己拿到了。

## 本地时间

`submit` **必须**带 `--local`：这份 kernel 在本地容器上每个点的 device 时间（µs），
逗号分隔，点 1 在前。少一个、非正数，都在占名额之前就拒掉。存在 `local_time` 表，
`local_vs_judge` 视图把它和判题器的时间按点对齐——可见点有 `judge_over_local`，
隐藏点判题器不返回时间所以是 null，但本地那一列留着。本地→OJ 是逐点映射而不是一个
全局系数，这张视图就是每次提交自动攒下的那组配对样本。

## 点数和可见性不是常量

点数、哪些点隐藏，都以判题器为准：排行榜回复里有 `total_testcase_count` 和
`testcases[]`（可见点带实时 `tbest`，隐藏点只有 `{_id, hidden: true}`），看 `testcase`
视图。`submit` 提交前拉一次排行榜，`--local` 的个数必须等于那时的点数——收榜前加点之后，
15 个就不够了。`visible` 只认"判题器给了时间的点"，隐藏均分按
`(N·总分 − Σ可见) / (N − 可见数)` 算，N 取这次结果的点数，全部可见时就没有隐藏均分。

`import` 补回来的历史提交没有本地时间：当时没测，不编。

## 命令

```bash
lab quota  -a ozongzi                     # 今天还剩几次、冷却还有多久（判题器口径）
lab submit kernel.asc -a ozongzi --reason "single-kernel v3" \
    --local 22.70,9.06,3.67,19.44,3.65,3.39,4.87,4.97,12.46,5.98,7.55,7.71,19.70,49.50,41.18
lab submit -   -a ozongzi --reason "..." --local …   # kernel 从 stdin 读
lab status [id]                           # id 可以是前缀或 judge_id；默认最近一个判完的
lab status [id] --refresh                 # 先 GET 一次判题结果和排行榜，绝不重新 POST
lab history                               # 所有机器的提交，最新 30 条
lab leaderboard -a ozongzi
lab import -a ozongzi                     # 把不在日志里的历史提交补回来
lab account -a ozongzi [--refresh] [--token]
lab sync                                  # 提交本机记录和凭证，pull --rebase，push
lab source <id>                           # 当时送出去的确切文本
lab minify kernel.asc                     # 和提交路径同一个 strip
lab schema                                # 表和视图
lab sql "<query>"                         # 只读，吐 JSON
```

`submit`、`import`、刷新 token 之后会自动 `sync` 一次；同步失败只警告，不影响结果，
记录已经在磁盘上，下次 `lab sync` 带走。`sync` 只提交 `lab/data` 和
`secrets/accounts` 两个路径，工作区里别的改动用 `--autostash` 原样放回；但 push 推的是
当前分支，所以分支上别的本地提交会一起推上去。

`submit` 的 kernel 参数写 `-` 就从 stdin 读，组装器可以直接管道进来，不用落临时文件；
管道里什么都没有时在占名额之前就报错。冷却中的 `submit` 直接排队等，判题轮询最长 900 s。
过线的源码去掉注释和空行，存下来和算 sha 的都是**发出去的那一版**。

## 视图

| 视图 | 是什么 |
| --- | --- |
| `exchange` | 日志里的每次往返，带响应体 |
| `verdict` | 每个提交判题器的最后一次回答 |
| `case_raw` / `case_result` | 每个 verdict 每个用例一行，后者带 `score` / `ratio` |
| `visible` | 判题器给了时间且确实跑起来的点（编译错误也占满槽位，`time` 是 0 不是 null） |
| `testcase` | 最新一次排行榜里的点：点数、是否隐藏、可见点的实时 tbest |
| `tbest_history` | 每一次排行榜快照里每个可见点的 tbest |
| `best_time` | 每个用例见过的历史最优 |
| `submission_full` | 我们的提交：账号、机器、状态、判题结果、本地时间是否齐全 |
| `local_vs_judge` | 本地时间对判题器时间，逐点 |
| `board` / `ranking_row` | 最近一次 / 每一次排行榜快照的每一行 |
| `board_split` | 每个队的总分反解出的隐藏点均分 |
| `poll_history` | 每次判题实际排了多久 |

```bash
# 每个用例的最优，对着 shape.md 里的 tbest 看
lab sql "select case_no, best_us, samples from best_time order by case_no"

# 本地→判题器的逐点比值，攒了多少样本
lab sql "select case_no, count(*) n, avg(judge_over_local) r from local_vs_judge
         where judge_over_local is not null group by case_no"

# 这次 401 到底说了什么
lab sql "select body from exchange where status=401 order by id desc limit 1"
```

## 打分

`case_score` 在我们**跑赢记录中的 best** 时给满分。原来 Python 的公式在 t/best = 1/1.5
处分母为零，再快就变负、被压成 0 分——突破的那一刻面板显示 0。分数对 t 单调递减，
所以 t ≤ best 就是满分。见 `score.rs` 的 `score_is_monotone_in_time`。

## 常驻监控其他队的新提交

```bash
lab watch -a ozongzi --once               # 检查登录和接口
lab watch -a ozongzi                      # 每 60 s 拉一次，JSON Lines 输出到 stdout
```

`new_submission` 的 `row` 是排行榜原始行；`baseline` / `poll_ok` 是心跳，`poll_error`
是故障。排除我们队。重启后从日志里的历史榜单重建已见提交 ID，不会把旧榜刷成新提交。
失败退避重试，最长 15 分钟。只报榜上新出现的提交 ID，轮询之间被覆盖的看不到。不会
自动登录或提交。

### 部署到 hz

服务器上没有 checkout，按同样的相对结构摆：

```bash
# 本机交叉编译（需 cargo-zigbuild、Zig 和 Rust Linux target）
rustup target add x86_64-unknown-linux-gnu
cargo zigbuild -p lab --release --locked --target x86_64-unknown-linux-gnu

# 服务器上一次性准备
sudo useradd --system --home-dir /opt/gmmlab --shell /usr/sbin/nologin gmmlab
sudo install -d -o gmmlab -g gmmlab -m 700 /opt/gmmlab /opt/gmmlab/bin \
     /opt/gmmlab/secrets /opt/gmmlab/secrets/accounts /opt/gmmlab/lab
# 把 secrets/accounts/ozongzi.json 拷到 /opt/gmmlab/secrets/accounts/，属主 gmmlab，600
sudo install -m 644 lab/deploy/gmmlab-watch.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now gmmlab-watch

# 更新二进制
scp target/x86_64-unknown-linux-gnu/release/lab hz:/opt/gmmlab/bin/lab.new
ssh hz 'chmod 755 /opt/gmmlab/bin/lab.new && mv /opt/gmmlab/bin/lab.new /opt/gmmlab/bin/lab \
        && systemctl restart gmmlab-watch'
ssh hz 'journalctl -u gmmlab-watch -f -o cat'
```

服务器上的日志在 `/opt/gmmlab/lab/data/`，不自动同步（那里没有 git checkout）；要带回来
就把 `lab/data/log/<hz 的机器名>/` 和 `body/` 拷进仓库再 `lab sync`，按机器分目录所以
不会和本地的冲突。旧版部署的库在 `/var/lib/gmmlab/gmmlab.db`，新版不读它。
