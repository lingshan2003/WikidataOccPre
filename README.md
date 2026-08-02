# Wikidata Occupation Prediction

基于人物知识图谱的职业分类实验。项目使用同一份已处理图数据，公平比较
R-GCN、R-GAT，以及未来可加入的 CompGCN。

## 唯一入口

所有新实验只使用 `run.py`：

```bash
python run.py prepare [prepare options]
python run.py train --model rgcn [training options]
python run.py train --model rgat [training options]
python run.py explain [explanation options]
```

旧根目录脚本仅为兼容保留；不要以 `main.py`、`train_rgat.py` 或旧
`data_loader.py` 作为新实验入口。

## 目录

```text
data/
├── extended.py           # 原始扩展 CSV -> 人物表、关系表
└── prepare.py            # 划分、特征、PyG graph_data.pt
models/
├── features.py           # 可扩展的类别/数值/向量属性编码与融合
├── rgcn.py               # 当前 R-GCN baseline
├── rgat.py               # 当前关系注意力模型
└── __init__.py           # 模型注册表；未来注册 CompGCN 的唯一位置
training/
├── train.py              # 共享 NeighborLoader 训练、评估与早停
└── explain.py            # R-GAT attention 候选边导出
legacy/original_baseline/ # 原始不可复现实验代码，只供追溯
run.py                    # 唯一命令入口
```

## 环境

服务器上先安装与 NVIDIA 驱动兼容的 PyTorch CUDA wheel 和匹配的 PyG
sampling wheels；随后安装项目其余依赖。具体组合见 PyG 官方 wheel 矩阵。

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --help
```

## 1. 准备数据

原始 `Q_R_Q_extended.txt` 含人物—关系—人物边及端点属性。预处理会创建
反向关系、稳定的节点级分层划分、Country 类别特征和出生/死亡时间特征。
职业只作为监督标签，不会输入模型，因而第一版没有目标标签泄漏。

```bash
python run.py prepare \
  --input Q_R_Q_extended.txt \
  --output-dir artifacts \
  --target-level 3 \
  --min-class-count 20 \
  --seed 42
```

生成的核心文件是 `artifacts/graph_data.pt`；`nodes.csv` 用于将预测和解释
映射回 Wikidata Q-ID。

## 2. 训练 R-GCN baseline

R-GCN 和 R-GAT 使用完全相同的数据、邻居采样、优化器和早停规则，因此结果
可直接比较。先运行它，得到真正的关系卷积基线。

```bash
python run.py train --model rgcn \
  --data artifacts/graph_data.pt \
  --output-dir runs/rgcn_level3 \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --num-bases 30 --rgcn-backend fast \
  --num-workers 4 --device cuda
```

`fast` 使用 `FastRGCNConv`，速度快但显存更高；若发生 OOM，可改为
`--rgcn-backend standard`，它较慢但更节省显存。

## 3. 训练 R-GAT

```bash
python run.py train --model rgat \
  --data artifacts/graph_data.pt \
  --output-dir runs/rgat_level3 \
  --epochs 50 --batch-size 512 --num-neighbors 15,10 \
  --hidden-dim 128 --branch-dim 64 --heads 4 \
  --num-workers 4 --device cuda
```

训练默认以 `val_loss` 选 checkpoint：patience 为 6，最小有效改善为 0.002；
验证 loss 连续 3 轮不改善时学习率减半。输出包括：

```text
runs/<experiment>/best_model.pt
runs/<experiment>/metrics.json
runs/<experiment>/test_predictions.csv
```

职业类别长尾时，可运行一个受控对照：在其余参数不变的前提下增加
`--class-weight`，比较 Macro-F1、Weighted-F1 与 Accuracy。

## 4. 导出 R-GAT 重要边候选

```bash
python run.py explain \
  --data artifacts/graph_data.pt \
  --checkpoint runs/rgat_level3/best_model.pt \
  --node-id Q1000023 \
  --output-dir explanations \
  --num-neighbors 15,10
```

输出包含该人物预测、特征融合 gate、两层 attention，以及指向该人物的 top
attention 边。Attention 仅代表模型的注意力候选，不是因果重要性；后续应对
top 边做删除实验，验证目标职业 logit 的下降。

## 扩展 CompGCN

在 `models/compgcn.py` 实现模型，接口保持：

```python
forward(features, edge_index, edge_type) -> logits
```

然后在 `models/__init__.py` 的 `MODEL_REGISTRY` 注册：

```python
"compgcn": RelationalCompGCNClassifier
```

之后无需改数据或训练代码，直接运行：

```bash
python run.py train --model compgcn ...
```
