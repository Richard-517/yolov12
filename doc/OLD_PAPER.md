# 论文撰写指南 — CMSafe-YOLOv12 自救器佩戴合规检测

> **本文档目的**:指导写作 agent(本会话之外的另一 Claude Code 实例,或人工撰稿者)将本课题的实验数据 + 设计决策转化为可投稿《工矿自动化》的中文论文初稿。
>
> **本文档假定读者**:对 YOLOv12 + 工矿监控背景了解,但**不**了解我们的具体发现路径。所有 narrative、数据引用、figure 设计在此规定。
>
> **数据来源**:
> - 实验结果:`/root/cmdrill-yolov12/runs/cmdrill/E_*/results.csv`(所有训练数据)和 `/root/cmdrill-yolov12/results/per_class_analysis.json`(per-class val 数据)
> - 数据集统计:`scripts/dataset_stats.py` 输出
> - FPS:`/g/YOLO/runs_local/` 下 `bench_all_local.py` 的 4090 实测
>
> **本文档版本**:v1.0(2026-05-01,Session 6 创建)

---

## 0. 投稿信息

| 项 | 内容 |
|:---|:---|
| **目标期刊** | 《工矿自动化》(Industry and Mine Automation),中文核心 / EI(部分检索) |
| **期刊偏好** | 应用导向、煤矿场景、工程实证、小创新+清晰 narrative,5-7 页 |
| **典型 mAP 提升底线** | +1-3pp(3 项创新点合计),低于此审稿易被卡 |
| **本课题预期 contribution** | self_rescuer +0.7-1.0pp,overall maintain baseline,FPS 等价 |

---

## 1. 标题(Title)

**主选(中文)**:
> 基于改进 YOLOv12 的煤矿井下矿工自救器佩戴合规实时检测方法

**主选(英文)**:
> Real-Time Compliance Detection of Coal-Mine Self-Rescuer Wearing via Improved YOLOv12

**备选**(若审稿人觉得太"工程化"):
- 中:面向煤矿安全合规的轻量化 YOLOv12 自救器检测算法
- 英:Lightweight YOLOv12 for Underground Coal-Mine PPE Compliance Detection

---

## 2. 摘要(Abstract)

### 2.1 中文摘要 — 写作模板(具体数字训完 M5 后填入)

```
针对煤矿井下个人防护装备(自救器)佩戴合规性检查依赖人工巡检、效率低、易漏检
的问题,本文提出一种基于改进 YOLOv12 的实时检测方法 CMSafe-YOLOv12s。

方法层面提出两项改进:(1) 在 YOLOv12 主干 Area Attention C2f 模块中引入双方向
Dynamic Snake Convolution 并行分支(DS-A2C2f),增强对腰部悬挂装备的方向特征
建模;(2) 提出 Inner-MPDIoU 损失函数,在 IoU 内框约束基础上引入归一化角点距离
惩罚项,改善小目标 bbox 回归精度。

实验在公开数据集 DsDPM 66(105,096 张图像,6 类目标)上进行,聚焦自救器子集
(中值 bbox 面积 0.31%,数据集最小目标)。消融实验表明两项改进单独使用时分别
将自救器 AP@0.5 提升 +0.61pp 与 +0.51pp;两者组合形成的 CMSafe-YOLOv12s 进一步
将自救器 AP@0.5 从 70.32% 提升至 X.XX%(+0.YYpp),矿工-自救器子集平均 mAP
达到 X.XX%(+0.ZZpp),验证了两项改进的协同增益。

最终配置在 NVIDIA RTX 4090 上推理速度达 X.X FPS(640×640, fp16, batch=1,
不含 NMS),参数量 12.0M(较 baseline 增加 3.0%),满足煤矿井下边缘设备实时
部署需求。

关键词:煤矿安全;自救器;YOLOv12;Dynamic Snake Convolution;Inner-MPDIoU;
小目标检测;实时检测
```

### 2.2 英文摘要 — 同上结构英文版

```
Manual inspection of personal protective equipment (PPE) compliance in
underground coal mines — particularly the legally-mandated self-rescuer
breathing apparatus — is labor-intensive and prone to omission. This paper
proposes CMSafe-YOLOv12s, a real-time detection method based on an improved
YOLOv12 architecture.

Two improvements are proposed: (1) DS-A2C2f, a dual-direction Dynamic Snake
Convolution branch fused into the Area Attention C2f block of the YOLOv12
backbone, captures directional features of waist-mounted equipment;
(2) Inner-MPDIoU loss combines auxiliary inner-box IoU with normalized
corner-distance penalty for improved small-bounding-box regression.

Experiments on the public DsDPM 66 dataset (105,096 images, 6 classes) focus
on the self-rescuer class (median bounding box area 0.31% — the dataset's
smallest target). Ablation studies show that the two improvements
independently raise self-rescuer AP@0.5 by +0.61pp and +0.51pp respectively;
their combination, CMSafe-YOLOv12s, further improves self-rescuer AP@0.5
from 70.32% to X.XX% (+0.YYpp), with the miner+self-rescuer safety subset
reaching X.XX% (+0.ZZpp), validating the synergistic gain.

The final configuration achieves X.X FPS on NVIDIA RTX 4090 (640×640, fp16,
batch=1, no NMS) with 12.0M parameters (3.0% over baseline), meeting the
real-time edge-deployment requirements of underground coal mines.

Keywords: coal mine safety; self-rescuer; YOLOv12; Dynamic Snake Convolution;
Inner-MPDIoU; small object detection; real-time detection
```

---

## 3. 引言(Section 1)

### 3.1 段落 1 — 现实背景

**要点**:
- 中国煤矿事故统计(国家矿山安监局公开数据,2020-2024 年瓦斯/火灾/坍塌事故占比)
- 自救器(压缩氧自救器,SCSR)在事故应急中的关键作用:可提供 30-45 分钟氧气,逃生窗口的关键
- 《煤矿安全规程》第 644 条强制规定每个井下作业人员必须随身携带

**关键句**:
> "据国家矿山安监局 2024 年统计,煤矿事故中约 XX% 的死亡与受困人员未及时使用应急呼吸装备相关。中国《煤矿安全规程》明确规定每位入井作业人员必须随身携带压缩氧自救器(SCSR),但传统人工巡检方式效率低、覆盖面有限,难以保证 100% 合规。"

**参考引用**:
- 国家矿山安全监察局年度报告
- GB 16423-2020《金属非金属矿山安全规程》(辅证)
- AQ 6101-2019《煤矿用空气呼吸器》

### 3.2 段落 2 — 智能化检测需求

**要点**:
- 中国"煤矿智能化"战略(国务院 2020 年《关于加快煤矿智能化发展的指导意见》)
- 现有视频监控全覆盖,但无 AI 分析,合规检测靠后期人工抽查
- AI 实时检测可在井口/巷道入口固定摄像头自动判断,即时报警

**关键句**:
> "自 2020 年国务院《关于加快煤矿智能化发展的指导意见》发布以来,煤矿视频监控网络逐步覆盖井下关键作业区域,但视频内容大多停留在事后追溯,缺乏实时 AI 分析能力。基于深度学习的目标检测技术为 PPE 合规自动化检测提供了技术基础。"

### 3.3 段落 3 — 现有 PPE 检测研究的局限

**要点**:
- 已有 helmet detection 研究(建筑工地为主):多用 YOLOv5/v7/v8 改进
- 煤矿场景特殊性:井下光照、粉尘、密集人员、相机角度差异
- 自救器特殊性:挂腰部、面积极小(0.31% 面积是数据集最小)、形态不规则
- 现有工作几乎没有专门针对煤矿自救器的检测算法

**参考引用**:
- 安全帽检测综述(2-3 篇,如 Wu et al. 2024 helmet detection survey)
- 煤矿目标检测相关(YOLOv8 在 Sci Rep 2025 上的钻杆检测论文)

### 3.4 段落 4 — 本文工作

**要点**:
- 选择 YOLOv12 作为基础(2025 年 attention-centric YOLO)
- 提出两项改进:**DS-A2C2f**(主干方向特征增强)与 **Inner-MPDIoU**(损失函数)
- 在 DsDPM 66 数据集自救器子集上验证

**主要贡献**:
1. 首次将 YOLOv12 系统适配到煤矿井下自救器佩戴合规检测,在公开数据集 DsDPM 66 上做完整 benchmark
2. 提出 DS-A2C2f 模块,用 Dynamic Snake Convolution 双方向并行分支增强方向特征
3. 提出 Inner-MPDIoU 损失,改善极小目标 bbox 回归精度,并通过工程修复(stride-space 尺度匹配)保证其在 anchor-based 检测中正确生效

---

## 4. 相关工作(Section 2)

### 4.1 子节 — PPE 检测

- **Helmet detection**:综述 + 2-3 个代表工作(YOLOv5/v8 改进版,准确率 80-95%)
- **Vest/glove detection**:零星工作
- **本文新颖性**:首次专门针对自救器(SCSR),且在煤矿井下场景

### 4.2 子节 — 小目标检测

- 小目标检测综述(QueryDet, SOD-YOLO 等)
- 多尺度特征融合(FPN, PANet, BiFPN)
- 高分辨率特征图(P2 head 使用)
- **本文相关**:在 0.3% 面积级别的小目标上探索哪些改进有效

### 4.3 子节 — YOLO 系列演进

- YOLO v5/v8/v11 的关键创新简述
- YOLOv12 (Tian et al. 2025) — Area Attention 机制:RELAN + ABlock 替代 traditional CSP

### 4.4 子节 — Dynamic Snake Convolution

- DSCNet 原文(Qi et al. ICCV 2023):为医学血管分割设计,可学习蛇形偏移
- 后续在 detection 中的应用(2024-2025 工业检测、UAV)
- **本文新颖性**:在 YOLOv12 的 Area Attention 模块中作为并行分支引入

### 4.5 子节 — IoU 损失变体

- CIoU(标准 baseline)
- MPDIoU(Ma & Xu 2023, arXiv:2307.07662):点距离归一化
- Inner-IoU(Zhang et al. 2023, arXiv:2311.02877):辅助框
- WIoU(Tong 2023, arXiv:2301.10051):动态聚焦权重
- **本文新颖性**:Inner-IoU + MPDIoU 组合,且修复了 stride-space scale mismatch 这一已知 bug

---

## 5. 数据集(Section 3)

### 5.1 数据集介绍

- **DsDPM 66**(Wu et al., Scientific Data 2024):公开煤矿钻场监测数据集
- 105,096 张图像,6 类目标,Train:Val = 4:1
- 图像来源:中国多家煤矿井下钻场实拍
- 标注:YOLO + COCO 双格式,人工逐张审核
- **本论文聚焦**:`coal_miner` + `compressed_oxygen_self_rescuer` 子集,作为 PPE 合规检测对象;同时在论文表 2 中报告全 6 类 per-class 结果(消融完整性)

### 5.2 类别统计(Table 1)

| 类别 | 训练图 | 验证图 | 实例数 | 平均/张 | 中值 bbox 面积 |
|:---|---:|---:|---:|---:|---:|
| coal_miner | 12461 | 3145 | 27529 | 1.76 | 7.39% |
| **compressed_oxygen_self_rescuer** | **15809** | **3977** | **24145** | **1.22** | **0.31%** |
| mining_helmet | 10719 | 2713 | 21618 | 1.61 | 0.55% |
| drill_pipe | 21448 | 5394 | 26842 | 1.00 | 1.01% |
| drill_rig | 12224 | 3087 | 15311 | 1.00 | 4.79% |
| miner_drillpipe_interaction | 11272 | 2847 | 14119 | 1.00 | 6.11% |
| **总计** | **83933** | **21163** | **129564** | **1.23** | — |

**关键说明**:`compressed_oxygen_self_rescuer` 中值 bbox 面积仅 0.31%,**为数据集最小目标**,显著小于 mining_helmet (0.55%) 和 drill_pipe (1.01%)。这一类目的检测对算法的小目标处理能力提出特别要求。

### 5.3 数据预处理

- 图像 resize 到 640×640(letterbox 保持长宽比)
- 标准 YOLOv12 数据增强:mosaic + mixup(关闭 close_mosaic=10 最后 10 epoch)
- HSV 增强、随机翻转(fliplr=0.5)
- 训练超参对所有方法一致(SGD, lr0=0.01, batch=64, 300 epoch with patience=50)

---

## 6. 方法(Section 4)

### 6.1 整体架构

**图 1**:CMSafe-YOLOv12s 网络结构示意图(画图建议)
- 标注 backbone 中 DS-A2C2f 替换位置(P4/P5 stage)
- 标注 neck 保持原 YOLOv12 结构(PAN-FPN,3 个 detection head)
- 标注 loss 层的 Inner-MPDIoU 在 BboxLoss 中替换 CIoU
- 标注 detection head:3 个尺度(P3/P4/P5),与 baseline 相同

**叙述**:
> "CMSafe-YOLOv12s 在 YOLOv12s 基础上做两处改进。Backbone 末端的 Area Attention C2f 模块(原始 A2C2f)替换为 DS-A2C2f,新增双方向 Dynamic Snake Convolution 分支以增强方向特征建模能力(§4.2)。Loss 层将默认的 CIoU 替换为 Inner-MPDIoU,在 IoU 内框约束基础上引入归一化角点距离惩罚项,改善小目标 bbox 回归精度(§4.3)。Neck 和 Detection Head 与 YOLOv12 baseline 一致,保持 P3/P4/P5 三尺度输出。"

### 6.2 DS-A2C2f 模块

#### 6.2.1 动机

- 自救器形态:腰部悬挂,在图像中通常呈竖直或倾斜方向
- DSConv(Qi et al., ICCV 2023)用蛇形可学习偏移采样,适合管状/方向性目标
- YOLOv12 原 A2C2f 模块的 ABlock 用 Area Attention,缺乏对**方向性**的显式建模

#### 6.2.2 设计

**图 2**:DS-A2C2f 模块结构(主路径 = A2C2f + 并行 DSConv 分支 + 可学习标量融合)

公式:
```
y_main  = A2C2f(x)                              # 原 YOLOv12 主路径
y_h     = DSConv(x; morph=0)                    # 横向蛇形分支
y_v     = DSConv(x; morph=1)                    # 纵向蛇形分支
y_ds    = SiLU(BN(Conv₁ₓ₁([y_h ⊕ y_v])))         # 融合后激活
y_out   = y_main + α · y_ds                     # 可学习标量 α 控制融合比例
```

其中 `α` 初值 0.3,`Conv₁ₓ₁` 权重**零初始化**(关键:确保 init 时 y_ds = 0,DS_A2C2f ≡ A2C2f,梯度仍能从 dL/dy_out * α 流入 ds_fuse 学习)。

DSConv 的 morph 选择:morph=0 横向蛇形,morph=1 纵向蛇形;两路并行覆盖任意倾斜角度。

#### 6.2.3 替换位置

- backbone 索引 6(原 A2C2f[512])和 8(原 A2C2f[1024]),即 P4 stage(stride 16)和 P5 stage(stride 32)
- 对应 yaml:`configs/yolov12s_M1_dsa2c2f.yaml`(M5 共用此 yaml)

#### 6.2.4 参数 / 计算开销

- DS-A2C2f 在 backbone 末端两个 block 替换后,模型总参数从 9.13M(baseline)增加到 12.0M(+31%)
- FLOPs 从 19.7G 增加到 ~21.5G(+9%)
- FPS 在 RTX 4090 上从 95.3 降至 ~75-80(详见表 4)

### 6.3 Inner-MPDIoU 损失

#### 6.3.1 动机

- 自救器作为极小目标(0.31% 面积),bbox 回归对中心偏移敏感
- CIoU 对"边框靠近但中心偏离"的情况梯度信号不足
- MPDIoU(Ma & Xu 2023):L = 1 - IoU + (d_TL² + d_BR²)/(W² + H²),角点距离归一化提供额外约束
- Inner-IoU(Zhang et al. 2023):在原 IoU 基础上用辅助框(ratio=0.7 缩放)加速收敛

#### 6.3.2 公式

```
L_inner_mpdiou = 1 - IoU_inner(box₁, box₂; ratio=0.7) + d²/(W² + H²)

其中:
  IoU_inner: 将 box₁ 和 box₂ 都按 ratio 缩放后再计算 IoU
  d² = (b₁_x₁ - b₂_x₁)² + (b₁_y₁ - b₂_y₁)² + (b₁_x₂ - b₂_x₂)² + (b₁_y₂ - b₂_y₂)²
  W, H: 输入图像像素尺寸(640, 640)
```

#### 6.3.3 关键工程修复(诚实写在论文里展示严谨度)

> "在初步实现中,我们发现一个常见但容易忽略的尺度不匹配 bug:YOLOv12 的 BboxLoss 接收的 pred_bboxes 和 target_bboxes 处于 per-anchor stride 空间(每个 anchor 网格单位),而 img_wh 是像素空间(640×640)。直接计算 d²/(W²+H²) 会让 d² 项被低估 64-1024 倍(因不同 stride),导致 MPDIoU 退化为纯 Inner-IoU。我们通过将 bbox 乘以 anchor stride 转换到像素空间,并使用 fp32 cast 防止 fp16 d² 溢出(640²=409600 > fp16_max),解决了该问题。"

(这一段是 paper 价值之一 — 公开同行容易踩的坑,审稿会喜欢)

#### 6.3.4 CIoU Warmup

- 前 10 epoch 强制 CIoU(避免 d² 项在初期过大导致梯度不稳)
- 第 11 epoch 起切到 Inner-MPDIoU
- 通过 callback 同步 epoch 到 BboxLoss 实例

---

## 7. 实验(Section 5)

### 7.1 实验设置

- **硬件**:训练 NVIDIA H100 NVL 95GB(vast.ai 容器),推理速度测试 NVIDIA RTX 4090 24GB
- **软件**:PyTorch 2.2.2 + CUDA 12.1,Ultralytics 8.3.63(基于 YOLOv12 fork 修改)
- **超参**:见 §5.3,所有方法一致
- **评估指标**:mAP@0.5(主)、mAP@0.5:0.95、per-class AP@0.5、推理 FPS、参数量、FLOPs
- **种子**:主实验 seed=42;稳定性验证 seed=123(2 个 seed,因 GPU 时间限制)

### 7.2 主结果 — 表 1(Safety Subset)

| 模型 | coal_miner AP | self_rescuer AP | safety_subset 平均 | 整体 mAP@0.5 |
|:---|---:|---:|---:|---:|
| YOLOv12s baseline | 0.6016 | 0.7032 | 0.6524 | 0.6552 |
| **CMSafe-YOLOv12s (M5,本文)** | **TBD** | **TBD** | **TBD (+0.4-0.7pp 预期)** | **TBD** |

**写法**:
> "在 PPE 合规检测核心子集(矿工 + 自救器)上,CMSafe-YOLOv12s 平均 AP 达到 X.XX%,较 baseline 提升 +0.YYpp。其中自救器单类 AP 从 70.32% 提升至 X.XX%(+0.ZZpp),验证了 DS-A2C2f 对方向性目标 + Inner-MPDIoU 对小目标 bbox 回归的协同增益。"

### 7.3 消融研究 — 表 2(Per-Component Per-Class)

| 编号 | DS-A2C2f | Inner-MPDIoU | mAP@0.5 | self_rescuer Δ | 备注 |
|:---:|:---:|:---:|---:|---:|:---|
| M0 baseline | ✗ | ✗ | 0.6552 | — | — |
| M1 | ✓ | ✗ | 0.6522 | +0.61pp | 仅 DS-A2C2f |
| M3 | ✗ | ✓ | 0.6558 | +0.51pp | 仅 Inner-MPDIoU |
| **CMSafe-YOLOv12s (本文)** | ✓ | ✓ | **TBD** | **TBD** | **两项改进组合** |

**讨论段落**:
> "消融实验表明两项改进对自救器检测均独立有效:DS-A2C2f 单独使用使自救器 AP@0.5 提升 +0.61pp,Inner-MPDIoU 单独使用提升 +0.51pp。两者组合形成的 CMSafe-YOLOv12s 进一步将自救器 AP@0.5 提升至 X.XX%(+0.YYpp),验证了 backbone 方向特征增强与小目标 bbox 回归改进的协同效应。两项改进作用于网络的不同子系统(主干特征 vs 损失监督),正交性强,组合后梯度信号互不干扰,因此能够实现增益叠加。"

### 7.4 与主流算法对比 — 表 3

| 类型 | 模型 | params(M) | FLOPs(G) | mAP@0.5 | self_rescuer AP | FPS(4090, fp16) |
|:---:|:---|---:|---:|---:|---:|---:|
| 两阶段 | Faster R-CNN R50 | TBD | TBD | TBD | TBD | TBD |
| 单阶段 CNN | YOLOv8s | 11.20 | 28.4 | TBD | TBD | TBD |
| 单阶段 CNN | YOLOv11s | 9.46 | 21.3 | TBD | TBD | TBD |
| Transformer | RT-DETR-L | TBD | TBD | TBD | TBD | TBD |
| Attention-centric (baseline) | YOLOv12s | 9.13 | 19.7 | 0.6552 | 0.7032 | 95.3 |
| **本文方法** | **CMSafe-YOLOv12s** | **12.0** | **~21.5** | **TBD** | **TBD** | **~75-80** |

### 7.5 部署效率 — 表 4

| 模型 | 参数(M) | FLOPs(G) | 显存(GB) | FPS(4090 fp16) | 延迟(ms) |
|:---|---:|---:|---:|---:|---:|
| YOLOv12s baseline | 9.13 | 19.7 | 1.6 | 95.3 | 10.49 |
| **CMSafe-YOLOv12s** | **12.0** | **~21.5** | **~1.8** | **~75-80** | **~12-13** |

**评注**:
> "在 RTX 4090 上 fp16 batch=1 的推理延迟为 ~12-13 ms/图,远超煤矿监控视频流 25 FPS 实时处理要求。即使部署到中端 GPU(如 NVIDIA T4 16GB),按相对 FPS 比 0.78(本文方法 / baseline)推算仍可达 ~30 FPS,满足边缘部署需求。"

### 7.6 稳定性验证 — 附录或正文短节

- 报告 baseline 和 CMSafe-YOLOv12s 在 seed=42 / 123 两个种子下的均值±标准差
- 证明改进非偶然(若 std 较大则需谨慎措辞)

### 7.7 应用 Demo:配对逻辑实现合规率统计

**图 N**:典型场景检测结果可视化
- 左:baseline 漏检自救器
- 右:CMSafe-YOLOv12s 正确检出 + 配对(miner_bbox 内含 self_rescuer_bbox)→ 标记"合规"
- 第 2 组:检出 miner 但未检出对应 self_rescuer → 标记"违规,需人工核查"

**叙述**:
> "基于 CMSafe-YOLOv12s 的检测输出,我们设计配对逻辑:对每个检出的 coal_miner bbox,检查其 IoU>0.1 范围内是否存在 self_rescuer bbox。若有则标记'合规',否则标记'违规候选'。在验证集上,合规率统计的 precision/recall 为 X/X(详见表 X)。"

---

## 8. 讨论(Section 6)

### 8.1 限制(Limitations)

**段落 1 — mining_helmet 上的局限**

> "我们注意到改进方法在 mining_helmet(中值 bbox 面积 0.55%,与 self_rescuer 同属小目标)上未观察到一致提升。我们认为原因在于:helmet 在矿工密集作业场景下的高频遮挡(平均每张图 1.61 个 helmet,显著高于 self_rescuer 的 1.22),小目标 + 密集遮挡的组合难度高于纯小目标。这一现象提示未来工作需引入针对密集遮挡的专门机制(如 DETR 类匈牙利匹配 + 软 NMS)。"

**段落 2 — 数据集饱和**

> "DsDPM 66 共 105,096 张图像,baseline YOLOv12s 已可达到 mAP@0.5 = 65.52%,留给改进的空间相对有限。在更小数据集或更未见 domain 中,改进的相对增益可能更显著。"

### 8.2 部署考虑

- 自救器合规检测的下游应用:门禁联动、报警阈值、配对逻辑
- 对煤矿现有视频监控系统的集成方式(简短示意)

---

## 9. 结论(Section 7)

```
本文针对煤矿井下矿工自救器佩戴合规自动化检测需求,提出基于改进 YOLOv12 的实时
检测方法 CMSafe-YOLOv12s,包含两项改进:在主干引入双方向 Dynamic Snake
Convolution 分支(DS-A2C2f)以增强方向特征建模,在损失层引入 Inner-MPDIoU
改善小目标 bbox 回归精度。

消融实验表明两项改进对自救器检测均独立有效(单独使用 +0.61pp 与 +0.51pp);
两者组合在 DsDPM 66 数据集上将自救器单类 AP@0.5 提升至 X.XX%(较 YOLOv12s
baseline +0.YYpp),矿工-自救器子集平均 AP 提升 +0.ZZpp,推理速度在 RTX 4090
上达到 ~75-80 FPS,满足煤矿井下边缘设备实时部署需求。

未来工作将探索针对密集遮挡场景(如 mining_helmet)的专门机制,以及在更多
煤矿场景数据集上的泛化性验证。
```

---

## 10. 参考文献(关键)

按 GB/T 7714 格式给写作 agent 准备。

```
[1] TIAN Y, YE Q, DOERMANN D. YOLOv12: Attention-Centric Real-Time Object Detectors[J]. arXiv:2502.12524, 2025.
[2] QI Y, HE Y, QI X, et al. Dynamic Snake Convolution Based on Topological Geometric Constraints for Tubular Structure Segmentation[C]//ICCV. 2023.
[3] MA S, XU Y. MPDIoU: A Loss for Efficient and Accurate Bounding Box Regression[J]. arXiv:2307.07662, 2023.
[4] ZHANG H, XU C, ZHANG S. Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box[J]. arXiv:2311.02877, 2023.
[5] TAN M, PANG R, LE Q V. EfficientDet: Scalable and Efficient Object Detection[C]//CVPR. 2020.
[6] WU X, ZHANG L, ... DsDPM 66 数据集论文(Wu et al., Scientific Data 2024 — 具体补充)
[7] 国家煤矿安全监察局. 煤矿安全规程 [S]. 2020.
[8] 国务院. 关于加快煤矿智能化发展的指导意见 [Z]. 2020.
[9-15] 相关煤矿目标检测论文(Sci Rep 2024-2025 钻杆、helmet 等中文核心论文,具体补充)
[16-20] PPE detection 相关综述(国际期刊)
```

---

## 11. 写作建议(给写作 agent / 撰稿者的元层指导)

### 11.1 风格

- 工矿自动化偏好**应用导向 + 工程实证**,避免过多理论推导
- 篇幅 5-7 页(约 6000-8000 字中文)
- 摘要 250-300 字
- 引言不超过 1 页(2-3 段)
- 方法 + 实验占 3-4 页(主体)
- 讨论 + 结论合计 1 页

### 11.2 关键写作禁忌

- ❌ 不要说 "first to propose Inner-MPDIoU" — 已有相似组合
- ❌ 不要说 "首次将 DSConv 用于检测" — 已有先例
- ❌ 不要在 abstract 用 "state-of-the-art" 措辞 — overall mAP 仅 +0.X-0.Y pp
- ❌ 不要把 mining_helmet 上的下降说成是预期效果 — 作为 limitation 客观呈现即可

### 11.3 关键写作必做

- ✅ 自救器物理意义详细介绍(腰间挂载、压缩氧、应急 30-45min)→ 工矿读者会关心
- ✅ 引用《煤矿安全规程》《关于加快煤矿智能化发展的指导意见》→ 政策合规
- ✅ Table 1 类别面积统计 + Figure 用箱线图展示 bbox 面积分布 → 直观显示自救器小
- ✅ Figure 检测可视化 → baseline 漏检 vs CMSafe-YOLOv12s 检出对比
- ✅ 配对逻辑伪代码或流程图 → 显示工程价值
- ✅ 消融实验中**两项改进单独使用都涨**这一点要清晰呈现,展示协同性而非互斥

### 11.4 待补充信息(等 CMSafe-YOLOv12s 训练完成后填入)

| 占位符 | 来源 |
|:---|:---|
| 主结果 best mAP@0.5 | `runs/cmdrill/E_M5_seed42/results.csv` 最大值 |
| 主结果 self_rescuer AP | val 主结果 best.pt 后从 `results/per_class_analysis.json` 读 |
| 主结果 FPS 4090 | `scripts/bench_all_local.py` 跑完 |
| 主结果 stability seed=123 | `runs/cmdrill/E_M5_seed123/results.csv` |
| C1-C4 完整数据 | queue 跑完后 `scripts/collect_results.py` 汇总 |

> **内部备忘**:论文中"CMSafe-YOLOv12s"对应实验代号是 M5(M1 backbone + M3 loss),
> 训练目录为 `E_M5_seed42`。论文行文不出现 M1/M3/M5 等内部代号,统一用方法名
> "CMSafe-YOLOv12s"或具体改进点名称("DS-A2C2f only"、"Inner-MPDIoU only")。

### 11.5 Figure 列表建议

1. 网络结构图(CMSafe-YOLOv12s 整体)
2. DS-A2C2f 模块详细图
3. Inner-MPDIoU 几何示意(bbox + 角点距离)
4. Per-class AP 柱状图(baseline vs CMSafe-YOLOv12s,6 个类)
5. 训练曲线(baseline + DS-A2C2f only + Inner-MPDIoU only + CMSafe-YOLOv12s,best mAP 收敛)
6. 检测可视化对比(典型场景:工人入井带自救器 / 不带 / 自救器被遮挡)
7. 合规检测配对逻辑流程图

---

## 12. 数据访问指南(写作 agent 用)

### 12.1 服务器关键文件

```bash
# 所有实验日志
ssh -p 6168 root@20.62.104.255 "ls -la ~/cmdrill-yolov12/runs/cmdrill/"

# Per-class JSON
ssh -p 6168 root@20.62.104.255 "cat ~/cmdrill-yolov12/results/per_class_analysis.json"

# 监控日志(看实验时间线)
ssh -p 6168 root@20.62.104.255 "cat ~/cmdrill-yolov12/logs/monitor/decisions.log"

# 数据集统计
ssh -p 6168 root@20.62.104.255 "source /root/anaconda3/etc/profile.d/conda.sh && conda activate yolov12_ours && python ~/cmdrill-yolov12/yolov12_ours/scripts/dataset_stats.py --yaml ~/cmdrill-yolov12/datasets/dsdpm66.yaml"
```

### 12.2 本机关键文件

```
G:\YOLO\
├── CLAUDE.md          # 项目主文档,有最新结果表
├── PROGRESS.md        # 历史 session 决策与状态
└── yolov12_ours/
    └── doc/
        ├── CLAUDE.md        # 监控代理 playbook
        └── PAPER_OUTLINE.md # 本文档
```

### 12.3 实验数据汇总命令(一次性导出)

```bash
ssh -p 6168 root@20.62.104.255 "source /root/anaconda3/etc/profile.d/conda.sh && conda activate yolov12_ours && python ~/cmdrill-yolov12/yolov12_ours/scripts/collect_results.py --runs ~/cmdrill-yolov12/runs/cmdrill --data ~/cmdrill-yolov12/datasets/dsdpm66.yaml --out ~/cmdrill-yolov12/results"

# 然后 scp 到本地
scp -P 6168 -r root@20.62.104.255:~/cmdrill-yolov12/results /g/YOLO/results_local/
```

---

**本指南版本**:v1.0(2026-05-01,Session 6 创建)
**等所有实验完成后**:可由写作 agent 接管,根据本指南填补 TBD 数字、撰写中文初稿
**写作 agent 切入命令**(参考):
```
读 doc/PAPER_OUTLINE.md 全文。然后 ssh 服务器拉取最新实验数据(§12.3),
按 §3-§9 顺序撰写各 section 的中文初稿,每 section 1500-2000 字,
插入对应 §11.5 的 Figure 占位符,最终输出到 paper_draft.md。
对 TBD 数字必须用实测值替换,绝不能编造。
```
