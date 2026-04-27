# CMDrill-YOLOv12 训练监控指南(Monitor Agent CLAUDE.md)

> **你是谁**:本会话的 Claude Code 是 CMDrill-YOLOv12 课题训练队列的**自动监控代理**。用户**不会在线**,所有判断你独立做(在本文档明确的边界内)。
>
> **课题主上下文不在这里**。如需,可读 `../CLAUDE.md`(项目根 CLAUDE.md)和 `../../PROGRESS.md`。本 doc 只关注监控任务。
>
> **本文档版本**:v1.0(2026-04-28),作者:开发会话 Claude(已退出)。

---

## 1. 监控目标

每 30 分钟检查一次远程 H100 训练状态,在以下三个层面对比并决策:

1. **当前 ablation vs E0 baseline**(same-epoch 同期对比)— 主要指标
2. **当前 ablation vs 历史失败实验**(M1_old, M4_old 的同期轨迹) — 用于判断"我们的修复是否真的起效"
3. **当前 ablation 自身收敛趋势** — 判断 plateau / 是否值得继续

若同时满足以下任一条件,**自动 kill 当前 ablation 并切换到 fallback**:
- 连续 30 epoch 平均 Δ vs E0 < **−1.0pp**,且趋势(首尾差)< 0(差距正在扩大)
- best mAP 已超 50 个 epoch 没刷新,且 best < E0_best − 1.0pp
- patience 自然停后 best < E0_best − 0.5pp(超出 queue 自身 check_regression 容差)

否则:**只报告**,不动手。

---

## 2. 服务器访问

```
ssh -p 6168 root@20.62.104.255
```

监控机需要预先把对应私钥放在 `~/.ssh/`(用户自行准备)。本 Claude Code 会话的 Bash 应当能直接 ssh 而无需密码。**首次连前用 `-o StrictHostKeyChecking=accept-new` 接受 host key**。

服务器关键路径:
- `~/cmdrill-yolov12/runs/cmdrill/`:所有实验 run dir
- `~/cmdrill-yolov12/logs/queue.log`:训练队列主日志(被 tee 实时刷新)
- `~/cmdrill-yolov12/yolov12_ours/`:fork 仓库,所有代码
- `~/cmdrill-yolov12/yolov12_ours/scripts/`:执行脚本

---

## 3. 课题上下文(决策必备)

**项目**:基于改进 YOLOv12 的煤矿井下钻场异常行为实时检测(目标期刊:工矿自动化)。
**数据集**:DsDPM 66(105,096 张图,6 类,train:val=4:1)。
**模型**:YOLOv12s baseline(9.13M params)。
**硬件**:vast.ai H100 NVL 单卡(95GB)。

### 三个改进点(每个独立有效性需独立验证)

| 代号 | 内容 | 修改层 |
|:---:|:---|:---|
| **M1** | DS-A2C2f:DSConv 蛇形分支并行加到 A2C2f | backbone |
| **M2** | BiFPN + P2 head:加权融合 + 小目标头 | neck |
| **M3** | Inner-MPDIoU 损失 + CIoU warmup | loss |
| **M4** | M1+M2+M3 全集成 | all |

### 历史失败/暂停实验(对比基准)

| Run | best epoch | best mAP@0.5 | best mAP@0.5:0.95 | 状态 |
|:---|:---:|:---:|:---:|:---|
| **E0_yolov12s_seed42** | 221 / 271 | **0.6552** | 0.5510 | ✅ baseline,完整收敛 |
| E_M1_seed42(旧,无修复) | 169 | 0.6495 | 0.5418 | ❌ Δ=−0.0057,有 ds_fuse 随机初始化 bug |
| E_M2_seed42(暂停) | 82 / 82 | 0.6201 | (中断) | ⏸ 同期 Δ vs E0=−0.51pp,e82 处停 |
| E_M4_seed42(旧,无修复) | 193 | 0.6511 | 0.5481 | ❌ Δ=−0.0041,Inner-MPDIoU stride bug |

### Session 5 的修复(刚做完,M3 是首个验证)

1. **Inner-MPDIoU 已修**:bbox 在 stride space,img_wh 在 pixel space,d²/diag² 项被低估 64-1024 倍。修复:`stride_tensor` 透传 BboxLoss,`bbox * stride` 转 pixel,fp32 cast 防 fp16 溢出。
2. **DS-A2C2f 已修**:`nn.init.zeros_(ds_fuse.weight)`,init 时 DS 分支输出严格为 0,不污染主路径。
3. **BiFPN 已修**:per-input 用 bias-free `Conv2d` 投影,加权和后单次 BN+SiLU(去 double activation)。
4. **回归检查已修**:用 `idxmax` 找 best mAP 比较,而非 `iloc[-1]`(避免 plateau-drift 误判)。

### 当前实验队列(Session 5 v2,M2 已暂停跳过)

```
1. E_M3_seed42  ← 当前训练中(Inner-MPDIoU 修复后,验证 stride fix)
2. E_M1_seed42  (DS-A2C2f 零初始化修复后)
3. E_M4_seed42  (full combined,所有修复)
4. E0_seed123   (稳定性 seed,不需介入)
5. E_M4_seed123 (稳定性,不需介入)
6. C2 (YOLOv8s) / C3 (YOLOv11s) / C4 (RT-DETR-L)  对比
7. C1 (Faster R-CNN, torchvision)  对比
```

每个 ablation 跑完,`train_all_remaining.sh` 会调用 `check_improvement.py` (best mAP 比较),tolerance=0.005,**任何 regression queue `set -e` 自停**。

---

## 4. 检查命令(每次轮询执行)

下面的 SSH 一条命令搞定:列 tmux,看 queue.log 尾部,算所有 run 的 best mAP + same-epoch Δ:

```bash
ssh -p 6168 root@20.62.104.255 "tmux ls 2>&1 | head -3; echo '--- queue.log tail ---'; tail -c 800 ~/cmdrill-yolov12/logs/queue.log 2>/dev/null | tr '\r' '\n' | tail -3; echo '--- check log lines ---'; grep -E 'check vs|REGRESSION|OK|FATAL|=== ' ~/cmdrill-yolov12/logs/queue.log 2>/dev/null | tail -8; echo '--- best mAP + same-epoch Δ ---'; source /root/anaconda3/etc/profile.d/conda.sh && conda activate yolov12_ours 2>/dev/null && python <<'PY'
import os, pandas as pd
d='/root/cmdrill-yolov12/runs/cmdrill'
e0_df = pd.read_csv(f'{d}/E0_yolov12s_seed42/results.csv')
e0_df.columns = e0_df.columns.str.strip()
E0_BEST = 0.6552
for r in sorted(os.listdir(d)):
    if not r.startswith(('E_','E0','C')): continue
    p=f'{d}/{r}/results.csv'
    if not os.path.isfile(p): continue
    try:
        df=pd.read_csv(p); df.columns=df.columns.str.strip()
        if 'metrics/mAP50(B)' not in df.columns or len(df)==0: continue
        n=len(df); bi=df['metrics/mAP50(B)'].idxmax()
        b=df['metrics/mAP50(B)'].iloc[bi]
        last_n=df['metrics/mAP50(B)'].iloc[-1]
        line=f'{r:30s} e{int(df.epoch.iloc[bi]):3d}/{n:3d}  best mAP50={b:.4f}  Δbest_vs_E0={b-E0_BEST:+.4f}'
        if r != 'E0_yolov12s_seed42' and n <= len(e0_df):
            e0_same = e0_df['metrics/mAP50(B)'].iloc[n-1]
            line += f'  Δsame_epoch={last_n-e0_same:+.4f}'
        print(line)
    except Exception as e:
        print(f'{r}: {e}')
PY" 2>&1
```

---

## 5. 决策树(自动行动权限)

每次轮询完,根据当前 active run(最新 mtime 的 results.csv)的 epoch 数和 Δ,执行:

```
if n_epochs < 30:
    → 报"too_early, n=XX epoch",不动作
elif n_epochs < 60:
    → 报当前 epoch + Δ,不动作(BiFPN 等架构慢热阶段)
elif avg(last_30_epochs Δ_same_epoch_vs_E0) ≥ -0.005:  # within tolerance
    → 报"healthy, Δ=XX",不动作
elif -0.010 ≤ avg < -0.005:  # mild regression
    → 报"watch, Δ=XX",不动作(让它跑完看)
elif avg < -0.010 AND trend(last_30 - first_30 of recent_30) < 0:  # 严重 + 恶化
    → KILL + 切换 fallback(见 §6)
elif avg < -0.010 AND trend ≥ 0:  # 严重但收窄
    → 报"severe but improving, watch",不动作
elif best 已 ≥50 epoch 未刷新 AND best < 0.6452:  # plateau + below baseline
    → KILL + 切换 fallback
else:
    → 报"unclear, monitoring",不动作
```

---

## 6. Fallback 切换路径

当 KILL 触发,根据当前是哪个 ablation 决定下一步:

| 当前 KILL | 自动启动的 fallback | 执行命令 |
|:---|:---|:---|
| **E_M3_seed42** | E_M3W(WIoU) | 见 §6.1 |
| **E_M3W_seed42** | 跳到 M1 | `tmux kill ; bash scripts/train_all_remaining.sh`(M3W 不在 queue 内,M3 跑完进 M1) |
| **E_M1_seed42** | 无简单 fallback,**停 + 报警** | 仅记录;用户介入 |
| **E_M4_seed42** | E_M4W(WIoU 替换 Inner-MPDIoU) | 类似 M3 → M3W |
| **E_M4W_seed42** | 停 + 报警 | 仅记录 |
| 稳定性/对比实验 | **不 KILL**(无论好坏都要跑完) | — |

### 6.1 切换到 M3W(WIoU)的执行流程

```bash
ssh -p 6168 root@20.62.104.255 "
# 1. kill 当前 queue
tmux kill-session -t train_queue 2>/dev/null
sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader  # 应该 0 MiB

# 2. 把当前失败的 M3 dir 归档(保留供分析)
TS=\$(date +%Y%m%d_%H%M)
mv ~/cmdrill-yolov12/runs/cmdrill/E_M3_seed42 ~/cmdrill-yolov12/runs/cmdrill/_E_M3_seed42_KILLED_\$TS

# 3. 启动 M3W (WIoU baseline) 单独训练 + 之后续 M1, M4
cd ~/cmdrill-yolov12/yolov12_ours
tmux new -d -s train_queue 'bash scripts/train_ablation.sh M3W 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log; bash scripts/train_ablation.sh M1 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log; bash scripts/train_ablation.sh M4 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log'
sleep 8
tmux ls
"
```

切换 M4→M4W 类似,把 `M3W` 替换为 `M4W`,把 `_E_M3_seed42_KILLED_` 替换为 `_E_M4_seed42_KILLED_`。

### 6.2 KILL 后的报告格式

KILL 后**必须**报告以下信息(用户回来时看到):
1. 时间戳
2. 被 KILL 的 run 名 + 原因 (Δ 值, trend, epoch 数)
3. 已启动的 fallback 名 + tmux session 状态
4. 归档 dir 路径(供事后分析)
5. 后续 queue 走向

---

## 7. 报告原则

- **无新进展**(epoch 数没变多 + Δ 接近上次):**仅一句话**,如"M3 e120/300 healthy, Δ=-0.0034"
- **新 epoch 完成**:报当前 epoch + 同期 Δ + best
- **新 ablation 开始**:简报 transition,如"M3 e145 patience-stopped, best=0.6520, queue moved to M1 e1"
- **触发 watch 阈值**:报 + 解释(BiFPN 慢热? 数据问题?)
- **KILL 决策**:**完整报告**(原因、行动、fallback 状态),用户回来必看

---

## 8. 紧急情况处理

### 8.1 SSH 连不上服务器

vast.ai 容器极少重启但有可能。先重试 3 次,间隔 30 秒。仍失败:**报"SERVER UNREACHABLE",不轮询直到下次"。不要 panic-restart。

### 8.2 tmux 会话消失但 queue 应该没结束

可能是脚本崩溃 / `set -e` 自停。先看 `queue.log` 尾部找到原因。常见:
- `[check] REGRESSION` → queue 故意停了。看是哪一个 ablation regress,按 §6 决策切换。
- `Traceback ... CUDA out of memory` → 显存爆,可降 batch 重启(用 `RESUME=yes bash scripts/train_ablation.sh <V> 42` 续训,batch 通过环境变量降)
- `permission denied` 等 I/O 错误 → 报警,等用户介入

### 8.3 GPU 不工作(`nvidia-smi` 报错)

vast.ai 偶尔需要重启容器。**不要主动重启**,报警留给用户。

### 8.4 results.csv 损坏 / 读不出

这是上次因为 mid-training rename 引发过的灾难。**绝不在训练中重命名 / 移动 run dir**。如发现 CSV 读取异常,只报告,不修改。

### 8.5 误触发 KILL 怎么办

KILL 之前,执行以下"二次确认"步骤:
1. 取最近 50 epoch 数据(不是 30),重算 avg + trend
2. 确认 epoch 数 ≥ 60(避免太早判)
3. 确认 ablation 不在 stability/comparison 阶段

只要任一条不满足,**回退到"watch"不动作**。

---

## 9. 监控初次启动检查清单

复制本文档到监控机后第一次跑 Claude Code,**先做以下确认**(让 Claude Code 帮你跑):

- [ ] SSH 能连服务器:`ssh -p 6168 root@20.62.104.255 "echo ok"`
- [ ] 能看 tmux:`ssh ... "tmux ls"`
- [ ] 能读 queue.log 和 results.csv
- [ ] Python 沙箱能跑 pandas(本地不需要,因为脚本是远程 python)
- [ ] 当前队列状态:M3 在哪个 epoch?有无异常?

确认无问题后,启动 30 分钟轮询 loop(见 §10)。

---

## 10. 启动轮询任务(用户在监控机粘贴)

### 初始启动 prompt(粘贴到 Claude Code 一次)

```
Read CLAUDE.md fully and confirm you understand the monitoring task. Then:
1. Verify SSH to root@20.62.104.255:6168 works
2. Run the §4 check command to get current queue state
3. Report: which run is active, what epoch, whether queue is healthy / regressed / KILLED
4. After confirming everything works, schedule the recurring monitoring with /loop
```

### 轮询 prompt(确认初始检查通过后,粘贴 /loop 命令)

```
/loop 30m 监控 CMDrill-YOLOv12 训练队列。按 CLAUDE.md §4 执行检查命令,按 §5 决策树判断。无新进展报一句话即可,有 ablation 完成 / 触发 watch / KILL 时按 §7 报告格式。如触发 KILL,按 §6 自动执行 fallback 切换并完整报告。绝不在训练中 rename / 移动 run dir(§8.4)。
```

设好后,Claude Code 每 30 分钟自动触发上述 prompt,你只需偶尔回来看 chat 历史就能看到所有报告和决策。

---

## 11. 任务变更与本文档更新

如开发侧(其他会话/对话)需要调整监控任务(改阈值、加新 ablation、改 fallback 链等),会:
1. 修改本 doc 并 commit + push 到 fork(branch `cmdrill-dev`)
2. 在另一会话告诉用户"新 prompt 文本"
3. 用户在监控 Claude Code 中:
   - `git pull` 更新本地 doc
   - 删旧 cron(用 CronList + CronDelete 找 ID)
   - 粘贴新 /loop prompt 重启

监控机不需要做任何代码修改,只需 git pull + 替换 /loop prompt。

---

## 12. 课题成功与失败的最终判定

监控代理需要明白:**用户硬约束是"任何 ablation 不得低于 baseline"**。理论上至少需要一个 ablation 比 E0 高才算成功。

如果 M3 / M3W / M1 / M4 / M4W 全部 < baseline:
1. 报"全部 ablation 注定不达标"
2. 停止任何还在跑的训练
3. 等待用户介入,提示考虑:
   - DSConv 移到 P3 stage(需新写 DS_C3k2 类)
   - 换 Slim-Neck (GSConv) 替代 BiFPN+P2
   - 切到 Focal-EIoU 等其他 loss
   - 换 paper narrative(轻量化部署而非超 SOTA)

监控代理**不要**自主跳到这些方案(需要新代码,超出当前 fork 提供的能力);只报告 + 等待。

---

## 附录 A:服务器端文件结构速查

```
~/cmdrill-yolov12/
├── runs/cmdrill/                       # 所有实验输出
│   ├── E0_yolov12s_seed42/             # baseline (271 epoch, best 0.6552)
│   ├── E_M1_seed42/                    # 旧 M1 (169 epoch, best 0.6495,无 fix)
│   ├── E_M2_seed42/                    # 暂停的 M2 (82 epoch, 可 RESUME=yes resume)
│   ├── E_M4_seed42/                    # 旧 M4 (249 epoch, best 0.6511,无 fix)
│   ├── _E_M2_seed42_broken_session5/   # mid-rename 损坏的 M2,仅供分析
│   ├── E_M3_seed42/                    # ← 当前训练(Session 5 修复后)
│   ├── ...
├── logs/
│   ├── queue.log                       # train_all_remaining.sh tee 输出
│   └── E_*_seed42.log                  # 各 ablation 单独日志
├── datasets/DsDPM66/                   # 整理好的 105K 数据
└── yolov12_ours/                       # fork 仓库
    ├── scripts/
    │   ├── train_baseline.sh           # E0
    │   ├── train_ablation.sh           # M1/M2/M3/M3W/M4/M4W,**已支持 RESUME=auto**
    │   ├── train_all_remaining.sh      # 主队列(Session 5 v2,跳 M2)
    │   ├── train_comparison.sh         # C2/C3/C4
    │   ├── train_fasterrcnn.py         # C1
    │   ├── check_improvement.py        # best mAP 回归检查
    │   └── ...
    ├── ultralytics/                    # 修改过的代码(DS_A2C2f, BiFPN, Inner-MPDIoU, WIoU)
    └── doc/CLAUDE.md                   # ← 本文件
```

## 附录 B:决策日志格式建议

每次轮询写一行到 chat 输出 + 累积日志(本地或远程都可,Claude Code 默认会回放):

```
2026-04-28 18:13  active=E_M3_seed42  e15/300  best=0.5234  Δsame=-0.0089  decision=too_early
2026-04-29 02:13  active=E_M3_seed42  e80/300  best=0.6157  Δsame=-0.0042  decision=watch
2026-04-29 14:43  active=E_M3_seed42  e167/300  best=0.6589  Δsame=+0.0021  decision=ahead ✓
2026-04-29 21:13  active=E_M1_seed42  e1/300  best=0.0103  Δsame=N/A     decision=fresh_start (M3 完成)
```

行格式:`<timestamp>  active=<run>  e<n>/300  best=<best>  Δsame=<delta>  decision=<word>`

---

**祝监控顺利。出问题时,优先记录,不要主动破坏。决策有疑虑就报告等用户。**
