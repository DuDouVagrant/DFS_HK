# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目目标

构建**鲁棒的图提示学习**（Robust Graph Prompt Learning）。具体而言：在 GNN 预训练 + Prompt Learning 的范式下，使下游节点分类任务在面对图结构扰动（如 Metattack 边污染）时仍保持稳定且可接受的分类性能。

## 当前阶段

### 2026-05-28 Meeting 10：RobustPrompt-T 大规模调参（已完成）

> ⚠️ **重要上下文（2026-06-03 补充）**：Meeting 10 所有实验均在 **2026-06-03 对齐论文之前的旧代码** 上运行。
> 旧代码的训练流程为 `add_muti_pt → filter_module 剪枝 → GNN₁ → τ_tune 剪枝 → GNN₂`（两阶段边剪枝），
> 与论文 GPromptShield 的 `add_muti_pt → GNN₁ → τ_tune 剪枝 → GNN₂`（单阶段 τ_tune）不一致。
> **2026-06-03 移除 filter_module、严格对齐论文后的代码尚未跑过实验。Meeting 10 的数据不可直接作为论文对齐代码的参考。**

在修复后代码上进行了系统的超参数搜索。**关键发现：RobustPrompt-T 当前实现未表现出预期鲁棒性，best clean 0.34 仍低于 GPPT 0.44，且受攻击后性能断崖式下跌。**

#### 调参历程

**第一阶段：最保守版本（全关正则/attention，仅扫 prompt_lr × pt_threshold）**

| 组别 | p_plus | attention | pt_threshold | prompt_lr | weight_mse/kl/constraint | Clean Acc |
|---|---|---|---|---|---|---|
| manual_1 | 开 | 关 | 0.0 | 0.001 | 全 0 | 0.2403±0.0049 |
| manual_2 | 开 | 开 | 0.0 | 0.001 | 全 0 | 0.1435±0.0346 |
| manual_3 | 关 | 关 | 0.0 | 0.001 | 全 0 | 0.1848±0.0070 |
| **GPPT 对照** | — | — | — | — | — | **0.4350±0.0000** |

结论：attention **有害** (0.24→0.14)，p_plus **有帮助** (0.18→0.24)，RPrompt-T 远落后于 GPPT。

**第二阶段：prompt_lr × pt_threshold 网格搜索（no_attention, 无正则）**

| prompt_lr | pt=0.0 | pt=0.05 | pt=0.1 | pt=0.2 |
|---|---|---|---|---|
| 0.0003 | 0.1907 | 0.1908 | 0.1887 | 0.1870 |
| 0.001 | 0.2403 | 0.2555 | 0.2582 | 0.2673 |
| 0.003 | 0.2803 | 0.2880 | 0.2845 | 0.2863 |
| 0.005 | 0.2723 | 0.2898 | 0.2558 | **0.2940** |

第一轮最好：lr=0.005, pt=0.20 → 0.2940±0.0139

**第三阶段：最好附近细搜**

lr ∈ {0.004, 0.005, 0.006, 0.008} × pt ∈ {0.15, 0.20, 0.25, 0.30}

细搜最好：lr=0.008, pt=0.30 → **0.3108±0.0323**（方差大）
更稳组合：lr=0.004, pt=0.25 → **0.2915±0.0033**

**第四阶段：加正则（weight_mse × weight_kl）**

基于两套 base 参数扫正则：
- A (lr=0.004, pt=0.25): best mse=0.1, kl=0.1 → 0.3230±0.0305
- B (lr=0.008, pt=0.30): best mse=0.1, kl=0.1 → **0.3427±0.0589**（最高但方差大）

**第五阶段：固定 stable 参数跑全部污染浓度**

```
prompt_lr=0.004, pt_threshold=0.25, weight_mse=0.1, weight_kl=0.1
no_attention, filter_mode=original
```

**第六阶段：污染图单独调参（no_attention, 无正则）**

| ptb | 最佳参数 | 结果 |
|---|---|---|
| 0.05 | lr=0.004, pt=0.0 | 0.2275±0.0259 |
| 0.1 | lr=0.008, pt=0.3 | 0.1832±0.0471 |

在污染图上加正则（weight_mse/weight_kl）**没有帮助**，KL 正则大多拉低性能。

#### 最终实验数据汇总

**GPPT Baseline（各污染浓度）**

| 污染浓度 | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|
| Accuracy | 0.4350 | 0.2790 | 0.0700 | 0.0740 | 0.0280 | 0.0350 |

**RobustPrompt-T — Stable 参数（全浓度统一）**

| 污染浓度 | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|---|---|
| Accuracy | 0.3213±0.0274 | 0.1867±0.0150 | 0.0883±0.0047 | 0.1037±0.0120 | 0.1557±0.0164 | 0.1562±0.0266 |

**RobustPrompt-T — 各浓度最优（per-ptb tuning）**

| 污染浓度 | 0.00 | 0.05 | 0.10 |
|---|---|---|---|
| Accuracy | 0.3427±0.0589 | 0.2275±0.0259 | 0.1832±0.0471 |

#### 调参核心结论

1. **attention 有害** — 开启后性能显著下降（0.24→0.14），应保持关闭
2. **p_plus 有益** — 20-token bank + learned combination 优于单 prompt
3. **高 lr 偏好高 pt_threshold** — lr=0.008 时 pt=0.3 最好，lr=0.004 时 pt=0.25 最好
4. **正则效果有限** — 仅在 clean 上有微弱提升，污染图上无帮助甚至有害
5. **Clean 最优参数 ≠ 污染图最优参数** — 无法用一组参数同时做好 clean 和 attacked
6. **RobustPrompt-T 当前未展现鲁棒性** — 污染图上性能断崖式下跌，0.05 污染下就从 0.32 跌到 0.19，与 GPPT 的 0.28 相比没有优势

### 2026-05-21 修复（GPromptShield 代码对齐论文）

参考 `reference/GPromptShield_修复与审查报告.txt`，对 RobustPrompt-T 完成了三项核心修复：

1. **实现 `out_detect_pt`** — 基于边两端节点 cosine similarity 识别 OOD 边
2. **修复 Attention 融合机制** — 改为 readout-token 模式，key_padding_mask 标记空 slot
3. **实现 τ_tune 动态边剪枝** — 两阶段 GNN forward：第一次 GNN → cosine 剪枝 → 第二次 GNN（2026-06-03 已移除 filter_module 初筛，对齐论文单阶段剪枝）

### 2026-05-29 修复（NaN Loss 问题）

在 Meeting 10 调参过程中发现并修复了三个问题：
1. **Double Softmax** — Answering head 的 `Softmax(dim=1)` 与 `CrossEntropyLoss` 内部 `log_softmax` 冲突 → 改为纯 `Linear`
2. **梯度裁剪** — KL/MSE 正则项导致梯度爆炸 → `clip_grad_norm_(max_norm=1.0)`
3. **UnboundLocalError** — loss 为 NaN 时 `test_acc` 未赋值 → 初始化 `test_acc = float('nan')`

### 已锁定的预训练配置

经过超参数网格搜索，**标准 GraphCL 预训练配置**已锁定为：

```
aug1=dropN  aug2=permE  ratio=0.3  lr=0.01  epochs=200
```

主权重文件：`pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth`

两个权重目录 `pre_trained_model/` 和 `pre_trained_model_raw/` 均已包含完整的 18 个 GraphCL 权重文件，覆盖全部网格组合：
- aug1 ∈ {dropN, permE, maskN} × aug2 ∈ {dropN, permE, maskN} × lr ∈ {0.005, 0.01}
- 均使用 GCN backbone，256 hidden dim，epochs=200
- 另各有 1 个 64 dim 的辅助权重（GraphCL 与 GraphMAE）

### 数据准备情况

`data_attack_fewshot/Cora/shot_5/1/Meta_Self/raw/` 下已具备全部 6 个污染浓度的攻击数据：

| ptb | 文件 | 大小 |
|---|---|---|
| 0.00 | Meta_Self_Cora_0.0.pt | 267KB |
| 0.05 | Meta_Self_Cora_0.05.pt | 278KB |
| 0.10 | Meta_Self_Cora_0.1.pt | 288KB |
| 0.15 | Meta_Self_Cora_0.15.pt | 298KB |
| 0.20 | Meta_Self_Cora_0.2.pt | 310KB |
| 0.25 | Meta_Self_Cora_0.25.pt | 323KB |

每个浓度对应的 `_idx_train.npy`、`_idx_val.npy`、`_idx_test.npy` 也齐全。**可以立即开始各浓度的 RobustPrompt-T 实验。**

## 核心文件路径

### 入口脚本

- `MyPretrain.py` — 预训练入口
- `MyTask.py` — 下游任务入口

### 核心模块 `prompt_graph/`

| 子模块 | 用途 | 修改频率 |
|---|---|---|
| `tasker/task.py` | BaseTask：初始化 GNN、Prompt、Optimizer | 高 |
| `tasker/node_task.py` | NodeTask：节点分类训练与评估 | 高 |
| `prompt/RobustPrompt_T.py` | RobustPrompt-T（GPromptShield）实现 | **最高** |
| `prompt/GPPTPrompt.py` | GPPT 实现（baseline） | 低 |
| `filters/filter_factory.py` | Filter 注册工厂 | 中 |
| `filters/neighbor_similarity_filter.py` | OriginalFilter / NeighborSimilarityFilter / HybridFilter | 高 |
| `utils/get_args.py` | 全部命令行参数定义 | 中 |
| `evaluation/RobustPromptTranductiveEva.py` | RobustPrompt-T 评估逻辑 | 中 |

### 数据目录

- `data_attack_fewshot/` — **指定 shot/split 的攻击数据**（`--specified` 模式实际加载的数据源）
- `data_pyg/` — 默认划分的攻击数据与干净数据
- `data_fewshot/` — few-shot 划分索引与 induced graph 缓存

### 权重与日志

- `pre_trained_model_raw/` — 预训练权重（主要使用）
- `logs/` — 实验日志，按 `{prompt_type}/` 组织

### 可忽略的目录/文件

- `data_attack_from_default_split/` — 默认划分的攻击数据，当前实验不使用
- `zcy_edge.py`、`exp_record.ipynb`、`figure_plot/` — 临时记录与作图
- `generate_few_shot_attack.sh`、`generate_few_shot_attack_batch.sh` / `generate_few_shot_attack.py` — 攻击生成脚本（已生成完毕）

## 常用命令（远程服务器实操）

远程服务器工作目录：`/home/tony/LnL/DFS_HK2`，conda 环境：`LnL2`，GPU：NVIDIA RTX 5090。

### 2026-06-03 忠于论文实验（纯 τ_tune，无 filter_module）

代码已对齐论文 GPromptShield：
- Training: `add_muti_pt → GNN₁ → τ_tune (cosine 剪枝) → GNN₂`
- Eval: `add_muti_pt → GNN`（不剪枝）
- filter_module 已注释，不参与任何边剪枝

**Round 1: 论文默认参数跑全浓度 baseline**

```bash
# 论文默认参数（p_plus=True, use_attention=True, cosine_constraint=True）
# prompt_lr 使用默认 0.01（与 GPF/GPF-plus 一致）
for ptb in 0.00 0.05 0.10 0.15 0.20 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/paper_attacked_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

**Round 2: no_attention + 论文默认参数**

```bash
for ptb in 0.00 0.05 0.10 0.15 0.20 0.25; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/paper_noatt_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

**Round 3: Filtering Tips 分选阈值调参（当前 TODO — 论文 Section 4.2 参数，尚未调过）**

论文 GPromptShield 的三个 Filtering Tips 控制哪些节点获得哪种 defense prompt：

| 参数 | 默认值 | 含义 | 影响 |
|------|--------|------|------|
| `--pt_sim_threshold` | 0.4 | 邻居平均 cosine 阈值 | ≤此值的节点 → sim_pt |
| `--pt_degree_threshold` | 2 | 度数阈值 | ≤此值的节点 → degree_pt |
| `--pt_out_detect_threshold` | 0.5 | 边 cosine 阈值 | ≤此值的边标记 OOD，端点 → out_detect_pt |

全组合 = 5 × 4 × 5 = 100 组，采用分阶段贪心策略（~20 组）：

**Phase 3a: 固定 deg=2, ood=0.5，扫 sim（论文其余默认，先跑 clean）**

```bash
for sim_t in 0.2 0.3 0.4 0.5 0.6; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    --pre_train_model_path './pre_trained_model_raw/Cora.GraphCL.GCN.256_hidden_dim.aug1_dropN.aug2_permE.lr_0.01.pth' \
    --task NodeTask --dataset_name Cora --preprocess_method none \
    --gnn_type GCN --prompt_type RobustPrompt-T --shot_num 5 --run_split 1 \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 1 2 3 4 5 \
    --filter_mode original --no_attention \
    --pt_sim_threshold ${sim_t} \
    > logs/RobustPrompt-T/ft_sim${sim_t}_clean_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

**Phase 3b: 固定 best sim + ood=0.5，扫 degree**

```bash
for deg_t in 1 2 3 5; do
  ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${deg_t} ...
done
```

**Phase 3c: 固定 best sim + best deg，扫 ood**

```bash
for ood_t in 0.3 0.4 0.5 0.6 0.7; do
  ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${BEST_DEG} \
      --pt_out_detect_threshold ${ood_t} ...
done
```

**Phase 3d: 取 Phase 3a-3c 最优组合，在 0.05 和 0.10 污染图上验证**

```bash
for ptb in 0.05 0.10; do
  CUDA_VISIBLE_DEVICES=0 nohup python MyTask.py \
    ... --pt_sim_threshold ${BEST_SIM} --pt_degree_threshold ${BEST_DEG} \
        --pt_out_detect_threshold ${BEST_OOD} \
    --attack_downstream --specified --attack_method Meta_Self-${ptb} \
    > logs/RobustPrompt-T/ft_best_${ptb}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
done
```

**Round 4: 基于最优 Filtering Tips + 最优训练参数，全浓度验证 0.00–0.25**

注意：如果 Round 3 结果不理想，可能需要先回到 prompt_lr × pt_threshold 的网格搜索（Round 3 旧版），
在论文对齐代码上重新找到好的训练参数基线，然后再调 Filtering Tips。

### 预训练（生成新的 GraphCL 权重）

### 预训练（生成新的 GraphCL 权重）

```bash
python MyPretrain.py --task GraphCL --dataset_name Cora --gnn_type GCN \
    --hid_dim 256 --num_layer 2 --epochs 200 --seed 56 --device 0 \
    --aug1 dropN --aug2 permE --lr 0.01
```

## 架构要点

### 两阶段流程

1. **Pretrain**：GNN backbone 在无标签图上自监督训练
2. **Downstream**：冻结 backbone，仅训练 prompt + 线性分类头（answering head），few-shot 设定

### Prompt 类型速查

- **Transductive**（全图，当前主线）：`GPPT`、`RobustPrompt-T`、`RobustPrompt-GPF`、`RobustPrompt-GPFplus`
- **Inductive**（k-hop 子图 batch）：`All-in-one`、`Gprompt`、`GPF`、`GPF-plus`、`RobustPrompt-I`

### RobustPrompt-T 防御机制（2026-06-03 对齐论文 GPromptShield）

四类防御 prompt（全部已实现）：
- `sim_pt` — 邻居平均相似度低的节点（csim <= 0.4）
- `degree_pt` — 低度节点（deg <= 2）
- `out_detect_pt` — OOD 边两端节点（edge cosine <= 0.5）
- `other_pt` — 其余节点的增强 prompt（`p_plus` 模式：20-token bank + 学习权重组合）

训练流程（`Tune` 方法，2026-06-03 对齐论文 GPromptShield Section 4.3）：
1. `add_muti_pt` — 为每个节点添加其对应的防御 prompt（论文 Filtering Tips 全用于节点分选，不做边过滤）
2. 第一次 GNN forward（全图，无预剪枝）→ 中间 node embedding
3. τ_tune 剪枝 — 基于中间 embedding 的 cosine similarity 过滤边（论文唯一边剪枝，Equation 15）
4. 第二次 GNN forward（剪枝后图）→ 最终 node embedding
5. Loss = CE + weight_mse × L_s + weight_kl × L_pt + weight_constraint × L_constraint

推理流程（`RobustPromptTranductiveEva`）：
1. `add_muti_pt` — 添加防御 prompt
2. GNN forward — 不做任何边剪枝（对齐论文：推理时不做 pruning）

训练剪枝 vs 推理不剪枝的原因（对齐论文 GPromptShield Section 4）：
- τ_tune 属于 Indirect Amplification 训练策略，作用对象是 prompt 参数的学习过程
- 训练时剪枝：迫使 prompt 在不依赖可疑边的前提下学习鲁棒节点表示（类似 Dropout 的正则化作用）
- 推理时不剪枝：训练好的 prompt 已具备鲁棒性，直接在全图上利用全部结构信息做分类
- 若推理时也剪枝，反而可能误删正常边、损失有用信息——论文从未在推理阶段做边剪枝

注：当前为 transductive（全图）设定而非论文的 inductive（子图）设定，以节省计算开销。论文 τ_tune 在逐子图的 G_inductive 上操作，本实现等效地在全图上操作。

自注意力融合（`use_attention=True`）：
- readout token 前置在 K 个 prompt slot 前
- MultiheadAttention(query=key=value=所有 slot)
- key_padding_mask 屏蔽空 slot
- 输出取 readout token 位置 → 融合后的最终 prompt

### 推荐超参数配置

**Clean 最优（高方差）**：
```
prompt_lr=0.008, pt_threshold=0.30, weight_mse=0.1, weight_kl=0.1
no_attention, p_plus=True, filter_mode=original
→ 0.3427±0.0589
```

**Clean 稳定（推荐主实验）**：
```
prompt_lr=0.004, pt_threshold=0.25, weight_mse=0.1, weight_kl=0.1
no_attention, p_plus=True, filter_mode=original
→ 0.3213±0.0274
```

**污染图 0.05 最优**：
```
prompt_lr=0.004, pt_threshold=0.0, weight_mse=0, weight_kl=0
no_attention, p_plus=True
→ 0.2275±0.0259
```

**污染图 0.1 最优**：
```
prompt_lr=0.008, pt_threshold=0.3, weight_mse=0, weight_kl=0
no_attention, p_plus=True
→ 0.1832±0.0471
```

### 已知问题与待办

1. **Train/Eval 剪枝已对齐论文（2026-06-03 修复）**
   - Training: add_muti_pt → GNN → τ_tune cosine 剪枝 → GNN（单阶段剪枝，对齐论文 Equation 15）
   - Eval: add_muti_pt → GNN（不剪枝，对齐论文推理流程）
   - 论文的 Filtering Tips（degree/similarity/OOD）全部用于节点分选 prompt，不用于边剪枝

2. **Transductive vs Inductive 设定差异**
   - 论文在 inductive（k-hop 子图 batch）设定下做 τ_tune
   - 当前实现在 transductive（全图）设定下做，以节省计算开销

3. **整体鲁棒性不足（注：Meeting 10 结果来自旧代码，filter_module 仍参与边剪枝）**
   - Clean 最优 0.34 仍低于 GPPT 0.44（在 filter_module 激活的旧代码上）
   - 0.05 污染下即从 0.32 跌至 0.19，GPPT 同期为 0.28
   - 2026-06-03 移除 filter_module 对齐论文后**尚未重新跑实验**
   - 当前 TODO：以忠于论文的代码（纯 τ_tune）重新建立 baseline

4. **正则项在污染图上的反效果**
   - weight_mse/weight_kl 仅在 clean 上有微弱正面效果
   - 污染图上加正则普遍拉低性能

### 扩展或修改组件时的定位

- 新增/修改 **GNN Backbone** → `prompt_graph/model/`，并在 `task.py:initialize_gnn()` 注册
- 新增/修改 **Prompt** → `prompt_graph/prompt/`，并在 `task.py:initialize_prompt()` 注册
- 新增/修改 **Filter** → `prompt_graph/filters/`，并在 `filter_factory.py` 注册

## 协作原则

1. **简约至上** — 不要过度抽象。三个相似行不急着提取函数。只改任务需要的部分。
2. **谋定后动** — 修改前先确认改动范围，用 TodoWrite 列出步骤。不确定时先问，不要猜测。
3. **敢于质疑** — 如果提示词或指令存在模糊之处，在动手前主动确认。如果观察到异常，主动指出。
4. **高可读性** — 变量名清晰、逻辑平铺直叙。不写注释解释"做了什么"，只在 WHY 不显而易见时注释。
5. **核心依赖** — PyTorch Geometric、deeprobust。不要引入新依赖。
