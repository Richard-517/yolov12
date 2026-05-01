# CMSafe-YOLOv12 训练监控指南(Monitor Agent CLAUDE.md)

> **你是谁**:本会话的 Claude Code 是 CMSafe-YOLOv12 课题(原 CMDrill,Session 6 已 pivot)训练队列的**自动监控代理**。用户**不会在线**,所有判断你独立做(在本文档明确的边界内)。
>
> **课题主上下文不在这里**。如需,可读 `../CLAUDE.md`(项目根 CLAUDE.md v5.0)和 `../../PROGRESS.md`。本 doc 只关注监控任务。
>
> **本文档版本**:v1.1(2026-05-01,Session 6 课题 pivot 后小幅更新)
>
> ### Session 6 关键变化(必读)
>
> 1. **课题 pivot**:论文从"煤矿钻场异常行为检测"转为"煤矿井下矿工自救器佩戴合规检测"。监控代理无需关心 narrative 细节,**仍按原有规则监控训练状态**。
> 2. **新增 M5 实验**:`E_M5_seed42` = M1 backbone + M3 loss(无 BiFPN+P2),是 paper 主推配置。M5 命令模板见 §6.2 新增"命令 F"(M5 失败的 fallback:目前**无简单 fallback,直接 HARD STOP**)。
> 3. **队列顺序更新**:`M4 → M5 → E0_s123 → E_M5_s123 → C2/C3/C4 → C1`。**E_M4_seed123 已跳过**(M4 不是论文主结果)。
> 4. **历史已完成**:E0/M3/M1 全部 PASS;E_M4 训练中(本次 pivot 后不作主推);E_M2 暂停在 e82。
> 5. 监控代理逻辑(§4 检查命令、§5 决策树、§6 KILL fallback)**无变化**。

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

## 6. Fallback 切换路径(**完整链表,绝不偏离**)

监控代理**只能执行下表里已经预定义的命令**,**不得**自主设计任何新 fallback、改 hyperparam、改 batch、动 YAML、写新模块。下表覆盖了所有授权的状态转移。

### 6.0 状态机(顺序流转)

```
M3      ─┬─ best ≥ 0.6502 → 进 M1                             [PASS, 走主线]
         ├─ KILL/regression → 进 M3W                            [LOSS-LEVEL FALLBACK]
         └─ M3W KILL/regression → 标记 LOSS_DEAD,跳到 M1       [放弃 loss 创新]

M1      ─┬─ best ≥ 0.6502 → 进 M4                             [PASS]
         └─ KILL/regression → 标记 BACKBONE_DEAD,跳到 M4       [放弃 backbone 创新]

M4      ─┬─ best ≥ 0.6502 → 进稳定性 + 对比阶段                 [PASS]
         ├─ KILL/regression → 进 M4W                            [LOSS-LEVEL FALLBACK]
         └─ M4W KILL/regression → 全部失败,执行 §6.4 全停        [HARD STOP]

稳定性 (E0_s123, E_M4_s123)
        └─ 仅在主 ablation 至少一个 PASS 时才跑;否则 §6.4 跳过

对比 (C1, C2, C3, C4)
        └─ 仅在主 ablation 至少一个 PASS 时才跑;否则 §6.4 跳过
```

### 6.1 KILL 后单步执行命令(查表用)

| 当前被 KILL | 下一步动作 | 执行命令(完整) |
|:---|:---|:---|
| **E_M3_seed42** | 启 M3W | §6.2 命令 A |
| **E_M3W_seed42** | 标记 LOSS_DEAD,启 M1 | §6.2 命令 B |
| **E_M1_seed42** | 标记 BACKBONE_DEAD,启 M4 | §6.2 命令 C |
| **E_M4_seed42** | 启 M4W | §6.2 命令 D |
| **E_M4W_seed42** | **全部失败,§6.4 hard stop** | §6.4 命令 E |
| **E_M5_seed42**(Session 6 新加) | **直接 §6.4 hard stop**(无 fallback,M5 是最后王牌) | §6.4 命令 E |
| 稳定性 / 对比 | **不 KILL**(让 patience 自然停) | — |

### 6.2 命令模板(复制-粘贴执行,不要改参数)

#### 命令 A:M3 KILL → 启 M3W

```bash
ssh -p 6168 root@20.62.104.255 "
tmux kill-session -t train_queue 2>/dev/null
sleep 5
TS=\$(date +%Y%m%d_%H%M)
mv ~/cmdrill-yolov12/runs/cmdrill/E_M3_seed42 ~/cmdrill-yolov12/runs/cmdrill/_E_M3_seed42_KILLED_\$TS
cd ~/cmdrill-yolov12/yolov12_ours
tmux new -d -s train_queue '
bash scripts/train_ablation.sh M3W 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log
bash scripts/train_ablation.sh M1  42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log
bash scripts/train_ablation.sh M4  42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log
'
sleep 8
tmux ls
"
```

#### 命令 B:M3W KILL → 跳到 M1(标记 LOSS_DEAD)

```bash
ssh -p 6168 root@20.62.104.255 "
tmux kill-session -t train_queue 2>/dev/null
sleep 5
TS=\$(date +%Y%m%d_%H%M)
mv ~/cmdrill-yolov12/runs/cmdrill/E_M3W_seed42 ~/cmdrill-yolov12/runs/cmdrill/_E_M3W_seed42_KILLED_\$TS
echo 'LOSS_DEAD' > ~/cmdrill-yolov12/.failed_components
cd ~/cmdrill-yolov12/yolov12_ours
tmux new -d -s train_queue '
bash scripts/train_ablation.sh M1 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log
bash scripts/train_ablation.sh M4 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log
'
sleep 8
tmux ls
"
```

#### 命令 C:M1 KILL → 跳到 M4(标记 BACKBONE_DEAD)

```bash
ssh -p 6168 root@20.62.104.255 "
tmux kill-session -t train_queue 2>/dev/null
sleep 5
TS=\$(date +%Y%m%d_%H%M)
mv ~/cmdrill-yolov12/runs/cmdrill/E_M1_seed42 ~/cmdrill-yolov12/runs/cmdrill/_E_M1_seed42_KILLED_\$TS
echo 'BACKBONE_DEAD' >> ~/cmdrill-yolov12/.failed_components
cd ~/cmdrill-yolov12/yolov12_ours
tmux new -d -s train_queue '
bash scripts/train_ablation.sh M4 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log
'
sleep 8
tmux ls
"
```

#### 命令 D:M4 KILL → 启 M4W

```bash
ssh -p 6168 root@20.62.104.255 "
tmux kill-session -t train_queue 2>/dev/null
sleep 5
TS=\$(date +%Y%m%d_%H%M)
mv ~/cmdrill-yolov12/runs/cmdrill/E_M4_seed42 ~/cmdrill-yolov12/runs/cmdrill/_E_M4_seed42_KILLED_\$TS
cd ~/cmdrill-yolov12/yolov12_ours
tmux new -d -s train_queue 'bash scripts/train_ablation.sh M4W 42 2>&1 | tee -a ~/cmdrill-yolov12/logs/queue.log'
sleep 8
tmux ls
"
```

### 6.3 PASS 后转移(无需 KILL,自然完成)

`train_all_remaining.sh` 已在 ablation 末尾自动调用 `check_improvement.py`,**通过则自动进下一阶段**,代理只需观察。代理**不需要**手动启动下一阶段,除非走 §6.2 的 KILL 路径。

### 6.4 全失败 HARD STOP(§关键!)

**触发条件**(满足任一即触发):
- E_M4_seed42 KILL 后 E_M4W_seed42 也 KILL → 三个改进均无效
- M1 KILL 且 M3 也 KILL 且 M3W 也 KILL → loss 改进 + backbone 改进均无效

**动作 — 命令 E:**

```bash
ssh -p 6168 root@20.62.104.255 "
tmux kill-session -t train_queue 2>/dev/null
sleep 5
echo \"\$(date +'%Y-%m-%d %H:%M:%S') HARD_STOP — all ablations failed, paper innovations confirmed ineffective\" >> ~/cmdrill-yolov12/logs/HARD_STOP.log
nvidia-smi --query-gpu=memory.used --format=csv,noheader
"
```

**报告**(强制完整):
1. 时间戳
2. 失败的 ablation 列表 + 各自 best mAP + Δ vs E0
3. 已经归档的 KILLED dir 路径
4. **已停止所有训练,GPU 已释放**(贴 `nvidia-smi` 输出确认)
5. **明确告知用户:三创新点无效,等待新方案,监控代理无权设计**

之后**继续每 30 分钟轮询但只做"GPU 使用监控"(不应该有任何训练在跑)**,如发现 GPU 又被占用(用户可能远程在做新实验),报告并继续观察。

---

## 6A. 监控代理的硬边界(BOUNDARIES — 严禁越界)

监控代理**只允许做**以下事:
1. 执行 §4 的检查命令
2. 按 §5 决策树判断
3. 执行 §6.2 / §6.4 中**完全字面**的命令模板(包括 mv 归档命名)
4. 执行 §10 的轮询任务
5. 写报告 / 写日志
6. 用 `RESUME=yes bash scripts/train_ablation.sh <V> 42` 续训(仅当容器重启等导致中断且本来 ablation 还没结论)

**严禁**:
- ❌ 修改 `train_ablation.sh` / `train_all_remaining.sh` / 任何 Python 代码
- ❌ 修改 hyperparameter(lr, batch, epochs, patience, mosaic, etc.)
- ❌ 创建新 YAML / 新模块 / 新 ablation 变体(M5, M6, M3X, etc.)
- ❌ 改 fallback 链顺序或决策阈值
- ❌ 删除任何已完成实验的 run dir(包括 `_KILLED_*` 归档)
- ❌ 自主"建议"或"实施"新方法(如把 DSConv 移到 P3、用 Slim-Neck、加 EIoU 等)
- ❌ 在不在表中的状态做"创造性"决策

如出现表内未覆盖的情形(数据集变化、新增 ablation、queue 顺序变更),代理**只报告 + 等待用户介入**。表是封闭的。

---

## 6B. GPU 节省规则

- 一旦触发 §6.4 HARD STOP,**绝不再启动任何训练**(直到用户明确介入)
- 一旦发现 ablation 在前 60 epoch 同期 Δ < -1.5pp 且 patience(50) 还有 100+ epoch 远,**应触发 KILL**(不要等 patience 自然停)
- 稳定性 seed (s123) 和对比实验(C1-C4)只在**至少一个 ablation 通过**时才跑;否则跳过(`train_all_remaining.sh` 在 v2 中通过 `set -e + check_improvement.py` 已经自动处理这种情况,代理只需不主动重启 queue)
- 单实验 patience=50 已经是上界,代理**不要把 patience 调高**让训练跑更久(看 §6A)

---

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
/loop 30m 监控 CMDrill-YOLOv12 训练队列。按 CLAUDE.md §4 执行检查命令,按 §5 决策树判断。每次轮询结束后**必须**按 §13.2 把单行追加到服务器 ~/cmdrill-yolov12/logs/monitor/monitor.log,把完整状态快照 overwrite 到 last_check.txt;触发 KILL/WATCH-warning/HARD_STOP 时还要按 §13.3 把详细事件 append 到 decisions.log。无新进展报一句话即可,有 ablation 完成 / 触发 watch / KILL 时按 §7 报告格式。如触发 KILL,按 §6.2 表格中字面对应的命令模板执行 fallback 切换(不得改动命令任何部分),并完整报告。绝不在训练中 rename / 移动 run dir(§8.4)。绝不自主设计新 ablation / 改超参 / 改决策阈值(§6A 边界)。如触发 §6.4 HARD STOP 条件,执行命令 E 全停训练,完整报告 §12 信息后等用户介入。
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

**用户硬约束**:任何 ablation 不得低于 baseline(±0.5pp 容差)。理论上至少需要一个 ablation 比 E0 高才算成功。

### 终态判定矩阵

| M3 / M3W | M1 | M4 / M4W | 终态 | 监控代理动作 |
|:---:|:---:|:---:|:---|:---|
| 任一 PASS | 任一 PASS | 任一 PASS | ✅ FULL SUCCESS | 进稳定性 + 对比阶段 |
| 任一 PASS | PASS | M4 失败 + M4W 失败 | ⚠️ PARTIAL | 触发 HARD STOP §6.4,M4 失败说明组合干扰,等用户介入 |
| 任一 PASS | KILL | 任一 PASS | ⚠️ PARTIAL | 进稳定性 + 对比,但 M1 失败需在论文里说明 |
| 全 KILL | KILL | 全 KILL | ❌ HARD STOP | 全停,**绝不**自主提下一步方案 |

### 监控代理在 HARD STOP 后**只做以下事**

1. 报告"三创新点经验证无效"
2. 列出每个 ablation 的最终 best mAP + Δ vs E0
3. 列出归档的 KILLED dir 路径(供用户事后分析)
4. 确认 GPU 已释放
5. **等待用户介入**

**严禁**(再次重复):监控代理不得提出"建议尝试 DSConv 移 P3 / Slim-Neck / Focal-EIoU / 换 narrative"等新方案。**这超出代理权限**。所有"重新设计"的事都由用户在新对话里和开发会话讨论。

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

## 13. 服务器端日志写入(必须做!跨会话续接)

监控代理**每次轮询都要在服务器上落盘日志**。理由:
- 开发会话(其他对话/未来 session)能直接 `ssh ... cat ...` 读监控历史,不用查监控 Claude 的 chat 回放
- 监控 Claude 与开发 Claude 解耦,任何一边崩了/换会话,另一边都还能继续工作

### 13.1 日志路径(三个文件,各司其职)

```
~/cmdrill-yolov12/logs/monitor/
├── monitor.log        # 每次轮询 append 一行(纯文本,chronological)
├── decisions.log      # 仅 KILL / WATCH-warning / HARD_STOP 时 append(重要事件,详细)
└── last_check.txt     # 每次轮询 overwrite 完整状态快照(供秒读)
```

服务器已预创建该目录 + 说明文件。代理每次写之前 `mkdir -p` 即可,防御性。

### 13.2 每次轮询的写日志步骤(强制)

在 §4 检查命令读完数据 + §5 决策完成后,**必须**追加以下 SSH 写入:

```bash
ssh -p 6168 root@20.62.104.255 "mkdir -p ~/cmdrill-yolov12/logs/monitor && \
cat > ~/cmdrill-yolov12/logs/monitor/last_check.txt <<EOF
timestamp: \$(date '+%Y-%m-%d %H:%M:%S %Z')
active_run: <RUN_NAME>
epochs: <N>/300
last_epoch_mAP50: <X.XXXX>
best_mAP50: <X.XXXX>
delta_best_vs_E0: <±X.XXXX>
delta_same_epoch: <±X.XXXX>
decision: <word>   # too_early | watch | ahead | continue | KILL | HARD_STOP
notes: <one-line context>
EOF
echo '\$(date \"+%Y-%m-%d %H:%M:%S\")  active=<RUN_NAME>  e<N>/300  best=<X.XXXX>  Δsame=<±X.XXXX>  decision=<word>' >> ~/cmdrill-yolov12/logs/monitor/monitor.log"
```

代理把 `<RUN_NAME>`、`<N>`、`<X.XXXX>` 等占位符替换成本次实测值再发出。

### 13.3 重要事件追加 decisions.log(仅 KILL / WATCH / HARD_STOP)

仅在 §5 决策为 `KILL` / `WATCH-warning`(同期 Δ < -0.5pp 但还没到 KILL 阈值)/ `HARD_STOP` 时,**额外** append 多行事件:

```bash
ssh -p 6168 root@20.62.104.255 "cat >> ~/cmdrill-yolov12/logs/monitor/decisions.log <<EOF
=== \$(date '+%Y-%m-%d %H:%M:%S %Z') ===
event: <KILL | WATCH | HARD_STOP>
active_run: <name>
epochs_completed: <N>
best_mAP50: <X.XXXX>
delta_best_vs_E0: <±X.XXXX>
delta_same_epoch_avg30: <±X.XXXX>
trend_30: <±X.XXXX>
threshold_triggered: <which §5 rule fired>
fallback_launched: <next run name | NONE for HARD_STOP>
killed_dir_archived: <path | none>
notes: <one-line>
EOF
"
```

### 13.4 开发会话(任何后续 Claude session)如何读这些日志

直接 SSH 命令(无需 git pull,无需 chat 历史):

```bash
# 秒级了解最新状态
ssh -p 6168 root@20.62.104.255 cat ~/cmdrill-yolov12/logs/monitor/last_check.txt

# 最近 50 次轮询(过去 ~25 小时)
ssh -p 6168 root@20.62.104.255 tail -50 ~/cmdrill-yolov12/logs/monitor/monitor.log

# 所有重要事件(KILL/HARD_STOP/WATCH)
ssh -p 6168 root@20.62.104.255 cat ~/cmdrill-yolov12/logs/monitor/decisions.log

# 三个一起,一次拉
ssh -p 6168 root@20.62.104.255 "
echo '=== last_check ==='; cat ~/cmdrill-yolov12/logs/monitor/last_check.txt 2>/dev/null
echo '=== last 30 polls ==='; tail -30 ~/cmdrill-yolov12/logs/monitor/monitor.log 2>/dev/null
echo '=== decisions ==='; tail -50 ~/cmdrill-yolov12/logs/monitor/decisions.log 2>/dev/null
"
```

### 13.5 日志大小管理

每次轮询 1 行 monitor.log 单行 ~120 字符,30 min 一次,7 天 = ~336 行 = ~40KB。无需 rotation。

如 last_check.txt 看着累积过大(误将旧内容拼接而非覆盖),代理用 `>` 而非 `>>` 重写。

### 13.6 失败场景

如 SSH 写日志失败(网络抖动等),代理**继续完成本次决策 + 报告 chat**,不重试无限循环。下次轮询会写入"上次写日志失败,本次正常"。

---

**祝监控顺利。出问题时,优先记录,不要主动破坏。决策有疑虑就报告等用户。**
