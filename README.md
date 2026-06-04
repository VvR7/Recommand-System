# RecommandSystem — KuaiRec 时序推荐实验

基于 [RecBole](https://recbole.io/) 框架，在 KuaiRec 2.0 数据集上的序列推荐对比实验。

## 当前结果 
### 主实验
| 方法 | GAUC | Recall@5 | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | NDCG@20 | 说明 |
|------|------|----------|-----------|-----------|--------|---------|---------|------|
| **🥇 SASRecF+T+C** | **0.8839** | **0.5228** | **0.6727** | **0.8154** | **0.3713** | **0.4199** | **0.4561** | +TimeBias +Contrastive |
| **🥈 SASRecF+T** | **0.8712** | **0.4891** | **0.6397** | **0.7890** | **0.3452** | **0.3940** | **0.4318** | +TimeBias |
| **🥉 SASRecF+C** | **0.8702** | **0.4991** | **0.6467** | **0.7906** | **0.3526** | **0.4004** | **0.4368** | +Contrastive |
| 4. SASRecF | 0.8509 | 0.4449 | 0.5955 | 0.7538 | 0.3062 | 0.3550 | 0.3950 | +item cat features |
| 5. SASRec | 0.8249 | 0.4200 | 0.5610 | 0.7094 | 0.2948 | 0.3404 | 0.3779 | ID-only 基线 |
| 6. BERT4Rec | 0.8217 | 0.3882 | 0.5274 | 0.6858 | 0.2662 | 0.3112 | 0.3511 | 双向对比 |
| 7. DeepFM+Feat | 0.7067 | 0.1636 | 0.2989 | 0.4825 | 0.0965 | 0.1399 | 0.1863 | FM+MLP +特征 |
| 8. DeepFM | 0.6758 | 0.1572 | 0.2843 | 0.4613 | 0.0936 | 0.1343 | 0.1789 | FM+MLP |
| 9. NeuMF | 0.6480 | 0.1297 | 0.2417 | 0.4118 | 0.0762 | 0.1120 | 0.1548 | GMF+MLP |
| 10. BPR | 0.6389 | 0.1436 | 0.2501 | 0.3951 | 0.0859 | 0.1201 | 0.1565 | 矩阵分解 |
| 11. ItemKNN | 0.5525 | 0.1413 | 0.2316 | 0.3432 | 0.0859 | 0.1149 | 0.1430 | 协同过滤 |

### 消融实验
| 实验 | GAUC | vs SASRecF | 累计增益 |
|------|------|-----------|---------|
| SASRec (ID-only) | 0.8249 | — | — |
| + item cat features (=SASRecF) | 0.8509 | — | +3.2% |
| SASRecF + TimeBias only | 0.8712 | +2.4% | — |
| SASRecF + Contrastive only | **0.8702** | **+2.3%** | — |
| SASRecF + TimeBias + Contrastive | 0.8839 | +3.9% | **+7.1%** (over SASRec) |
## 环境配置

### 安装依赖

```bash
pip install recbole

# 若自动安装的 torch 版本过低/过高：
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# NumPy 兼容性
pip install "numpy<2.0.0"
```

### Patch 源码

将本仓库 `./model` 目录下的自定义模型文件复制到 RecBole 安装路径：

```bash
# 示例路径（根据实际环境调整）
RECBOLE_PATH=/home/david/miniconda3/envs/recbole/lib/python3.10/site-packages/recbole/model

# 复制自定义模型
cp model/sequential_recommender/bert4recf.py  $RECBOLE_PATH/sequential_recommender/
cp model/sequential_recommender/bert4recp.py  $RECBOLE_PATH/sequential_recommender/
cp model/sequential_recommender/sasrecp.py    $RECBOLE_PATH/sequential_recommender/

# 复制修改后的 layers.py（新增 RoPE + TimeBias + **kwargs 透传）
cp model/layers.py  $RECBOLE_PATH/layers.py

# 修复 NumPy 兼容性
运行代码时如果报错np.long，就改为np.int64
```
## 数据集准备

### 数据来源

KuaiRec 2.0 数据集（需手动下载到 `KuaiRec/` 目录）https://github.com/chongminggao/KuaiRec ：

```
KuaiRec/
├── KuaiRec/
│   └── KuaiRec2.0/
│       └── data/
│           ├── big_matrix.csv
│           ├── small_matrix.csv
│           └── user_features.csv
└── kuairec_caption_category.csv
```

### Step 1: 创建交互数据 (kuairec.inter)

```bash
# 在 id_only_baseline/scripts/ 中
python preprocess.py
```
输出：`dataset/kuairec/kuairec.inter`（4.17M 交互，5,765 用户，3,310 物品）

### Step 2: 创建特征数据 (kuairec.user + kuairec.item)

```bash
python preprocess_features.py
```
输出：
- `dataset/kuairec/kuairec.user` — 12 用户特征字段
- `dataset/kuairec/kuairec.item` — 3 物品类别特征字段（L1/L2/L3）

### 数据集文件最终结构

```
dataset/kuairec/
├── kuairec.inter          # 交互数据 (115 MB)
├── kuairec.user           # 用户特征 (408 KB)
├── kuairec.item           # 物品类别特征 (55 KB)
├── test_data.pkl          # 评测数据 (3.7M 行)
└── val_data.pkl           # 验证数据 (934K 行)
```

## 模型总览

### 传统方法（本机可跑）

| 模型 | 配置文件 | 说明 |
|------|---------|------|
| ItemKNN | `configs/itemknn_kuairec.yaml` | 基于物品的协同过滤 |
| BPR | `configs/bpr_kuairec.yaml` | BPR-MF 矩阵分解 |
| NeuMF | `configs/neumf_kuairec.yaml` | 神经矩阵分解 |
| DeepFM | `configs/deepfm_kuairec.yaml` | FM+MLP (ID-only) |
| DeepFM_Feat | `configs/deepfm_feat_kuairec.yaml` | FM+MLP + 用户/物品特征 |

### 序列模型（需服务器 >16GB CPU RAM）

| 模型 | 配置文件 | 说明 |
|------|---------|------|
| **SASRec** | `configs/sasrec_kuairec.yaml` | 单向 Transformer, ID-only |
| **SASRecF** | `configs/sasrecf_feat_kuairec.yaml` | SASRec + 物品类别特征 |
| **SASRecP** | `configs/sasrecp_kuairec.yaml` | SASRecF + RoPE + TimeBias + Contrastive |
| BERT4Rec | `configs/bert4rec_kuairec.yaml` | 双向 Transformer, ID-only |
| BERT4RecF | `configs/bert4recf_feat_kuairec.yaml` | BERT4Rec + 物品类别特征 |

### SASRecP 创新模块

SASRecP 包含三个可独立开关的创新模块，均在 `configs/sasrecp_kuairec.yaml` 中配置：

| 模块 | 配置开关 | 原理 |
|------|---------|------|
| RoPE | `use_rope: true/false` | 旋转位置编码，替换绝对位置 embedding |
| TimeBias | `use_time_bias: true/false` | 真实时间间隔作为 attention bias |
| Contrastive | `use_contra_loss: true/false` | 同类别物品对比学习辅助 loss |


## 训练

所有模型使用统一训练入口 `train_cf.py`：

```bash
python train_cf.py --model <模型名> [--config <配置文件>] [--smoke] [--device cuda]
```

### 传统方法（本机）

```bash
python train_cf.py --model ItemKNN
python train_cf.py --model BPR
python train_cf.py --model NeuMF
python train_cf.py --model DeepFM
python train_cf.py --model DeepFM --config configs/deepfm_feat_kuairec.yaml  # +特征
```

### 序列对比方法（服务器）

```bash
# BERT4Rec 系列
CUDA_VISIBLE_DEVICES=X python train_cf.py --model BERT4Rec
CUDA_VISIBLE_DEVICES=X python train_cf.py --model BERT4RecF --config configs/bert4recf_feat_kuairec.yaml

# SASRec 系列
CUDA_VISIBLE_DEVICES=X python train_cf.py --model SASRec
CUDA_VISIBLE_DEVICES=X python train_cf.py --model SASRecF --config configs/sasrecf_feat_kuairec.yaml
```

### SASRecP（主实验，服务器）
如果`--config_dict `报错，就删去`--config_dict`，去yaml文件改config

```bash
# 全开（三个模块都启用）
python train_cf.py --model SASRecP --config configs/sasrecp_kuairec.yaml

# 基线（等价于 SASRecF，验证无退化）
python train_cf.py --model SASRecP \
    --config configs/sasrecp_kuairec.yaml \
    --config_dict '{"use_rope":false,"use_time_bias":false,"use_contra_loss":false}'

# 仅 RoPE
python train_cf.py --model SASRecP \
    --config configs/sasrecp_kuairec.yaml \
    --config_dict '{"use_rope":true,"use_time_bias":false,"use_contra_loss":false}'

# 仅 TimeBias
python train_cf.py --model SASRecP \
    --config configs/sasrecp_kuairec.yaml \
    --config_dict '{"use_rope":false,"use_time_bias":true,"use_contra_loss":false}'

# 仅 Contrastive
python train_cf.py --model SASRecP \
    --config configs/sasrecp_kuairec.yaml \
    --config_dict '{"use_rope":false,"use_time_bias":false,"use_contra_loss":true}'

# RoPE + TimeBias
python train_cf.py --model SASRecP \
    --config configs/sasrecp_kuairec.yaml \
    --config_dict '{"use_rope":true,"use_time_bias":true,"use_contra_loss":false}'

# RoPE + Contrastive
python train_cf.py --model SASRecP \
    --config configs/sasrecp_kuairec.yaml \
    --config_dict '{"use_rope":true,"use_time_bias":false,"use_contra_loss":true}'

# TimeBias + Contrastive
python train_cf.py --model SASRecP \
    --config configs/sasrecp_kuairec.yaml \
    --config_dict '{"use_rope":false,"use_time_bias":true,"use_contra_loss":true}'
```

## 评测

统一评测入口 `evaluate_temporal.py`（时序留一法，80% 历史 → 20% 评测目标，无数据泄漏）：

```bash
python evaluate_temporal.py --checkpoint <checkpoint路径> --model <模型名> --split test
```

### 所有模型评测命令

```bash
# 传统方法
python evaluate_temporal.py --checkpoint saved/ItemKNN-*.pth --model ItemKNN --split test
python evaluate_temporal.py --checkpoint saved/BPR-*.pth --model BPR --split test
python evaluate_temporal.py --checkpoint saved/NeuMF-*.pth --model NeuMF --split test
python evaluate_temporal.py --checkpoint saved/DeepFM-*.pth --model DeepFM --split test
python evaluate_temporal.py --checkpoint saved/DeepFM-*.pth --model DeepFM_Feat --split test

# BERT4Rec 系列
python evaluate_temporal.py --checkpoint saved/BERT4Rec-*.pth --model BERT4Rec --split test
python evaluate_temporal.py --checkpoint saved/BERT4RecF-*.pth --model BERT4RecF --split test

# SASRec 系列
python evaluate_temporal.py --checkpoint saved/SASRec-*.pth --model SASRec --split test
python evaluate_temporal.py --checkpoint saved/SASRecF-*.pth --model SASRecF --split test

# SASRecP（无论训练时开了几个模块，命令相同——scorer 自动适配）
python evaluate_temporal.py --checkpoint saved/SASRecP-*.pth --model SASRecP --split test
```

## 项目结构

```
RecommandSystem/
├── configs/                  # 模型配置文件（YAML）
├── dataset/kuairec/          # 预处理后的原子文件 + 评测数据
├── model/                    # 自定义 RecBole 模型（需复制到 pip 路径）
├── log/                      # 训练/评测日志
├── saved/                    # 模型 checkpoint
├── result/                   # 评测结果
├── train_cf.py               # 统一训练入口
├── evaluate_temporal.py      # 时序留一法评测
├── preprocess_features.py    # .user / .item 生成
├── 当前完成情况.md            # 和ai交互时维护的文档
└── 实验计划.md                # 和ai交互时维护的文档
```



