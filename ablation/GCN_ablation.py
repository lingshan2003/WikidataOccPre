import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.nn import RGCNConv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import copy
#run通了，结果总体是加上educated at效果提升最好，训练每个epoch耗时3h
print("\nGPU Information:")
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda')
    # 清理GPU缓存
    torch.cuda.empty_cache()
    # 打印初始GPU内存使用情况
    print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Initial GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
else:
    print("CUDA is not available. Using CPU instead.")
    device = torch.device('cpu')

# 训练参数设置
num_epochs = 5
batch_size = 512  # 每次预测512个节点的职业
hidden_dim = 64
print_every = 1
batches_per_epoch = 100

try:
    # 加载数据
    edges = pd.read_csv('Q_R_Q.txt', sep=' ', header=None, names=['Q_node1', 'r', 'Q_node2'])
    attributes = pd.read_csv('filtered_Q_attribute(60.8w).txt', sep='\t')

    # 获取职业信息（所有level）
    edges['Q_node1_level1_main_occ'] = edges['Q_node1'].map(attributes.set_index('Q_node')['level1_main_occ'])
    edges['Q_node2_level1_main_occ'] = edges['Q_node2'].map(attributes.set_index('Q_node')['level1_main_occ'])
    edges['Q_node1_level2_main_occ'] = edges['Q_node1'].map(attributes.set_index('Q_node')['level2_main_occ'])
    edges['Q_node2_level2_main_occ'] = edges['Q_node2'].map(attributes.set_index('Q_node')['level2_main_occ'])
    edges['Q_node1_level3_main_occ'] = edges['Q_node1'].map(attributes.set_index('Q_node')['level3_main_occ'])
    edges['Q_node2_level3_main_occ'] = edges['Q_node2'].map(attributes.set_index('Q_node')['level3_main_occ'])
    
    # 获取work location信息
    edges['Q_node1_work_location'] = edges['Q_node1'].map(attributes.set_index('Q_node')['work location'])
    edges['Q_node2_work_location'] = edges['Q_node2'].map(attributes.set_index('Q_node')['work location'])

    # 获取新增特征：educated at、gender、country
    edges['Q_node1_educated_at'] = edges['Q_node1'].map(attributes.set_index('Q_node')['educated at'])
    edges['Q_node2_educated_at'] = edges['Q_node2'].map(attributes.set_index('Q_node')['educated at'])
    edges['Q_node1_gender'] = edges['Q_node1'].map(attributes.set_index('Q_node')['gender'])
    edges['Q_node2_gender'] = edges['Q_node2'].map(attributes.set_index('Q_node')['gender'])
    edges['Q_node1_country'] = edges['Q_node1'].map(attributes.set_index('Q_node')['country'])
    edges['Q_node2_country'] = edges['Q_node2'].map(attributes.set_index('Q_node')['country'])

    # 删除职业信息缺失的行
    print(f"原始边数量: {len(edges)}")
    edges = edges.dropna(subset=['Q_node1_level1_main_occ', 'Q_node2_level1_main_occ'])
    print(f"删除职业信息缺失后的边数量: {len(edges)}")

    # 编码节点和关系（只编码一次）
    remaining_nodes = pd.concat([edges['Q_node1'], edges['Q_node2']]).unique()
    remaining_relations = edges['r'].unique()

    node_encoder = LabelEncoder().fit(remaining_nodes)
    relation_encoder = LabelEncoder().fit(remaining_relations)

    edges['Q_node1_encoded'] = node_encoder.transform(edges['Q_node1'])
    edges['Q_node2_encoded'] = node_encoder.transform(edges['Q_node2'])
    edges['r_encoded'] = relation_encoder.transform(edges['r'])

    # 获取关系和节点数量（全局变量）
    num_relations = len(relation_encoder.classes_)
    num_nodes = len(node_encoder.classes_)

    # 为每个level创建职业编码器
    # 职业特征需要'unknown'标记，因为测试节点的职业信息会被mask为'unknown'
    occ_encoders = {}
    for level in [1, 2, 3]:
        level_occupations = sorted(set(
            pd.concat([
                edges[f'Q_node1_level{level}_main_occ'],
                edges[f'Q_node2_level{level}_main_occ']
            ]).unique().tolist() + ["unknown"]
        ))
        occ_encoders[level] = LabelEncoder().fit(level_occupations)
    
    # 创建work location编码器（合并所有节点的特征）
    # 只有职业特征需要'unknown'标记（被mask），其他特征只需要'missing'标记
    all_work_locations = pd.concat([
        edges['Q_node1_work_location'].fillna('missing'),
        edges['Q_node2_work_location'].fillna('missing')
    ]).unique()
    work_locations = sorted(set(all_work_locations.tolist() + ['missing']))
    work_location_encoder = LabelEncoder().fit(work_locations)

    # 创建新增特征编码器：educated_at、gender、country（合并所有节点的特征）
    all_educated_ats = pd.concat([
        edges['Q_node1_educated_at'].fillna('missing'),
        edges['Q_node2_educated_at'].fillna('missing')
    ]).unique()
    educated_ats = sorted(set(all_educated_ats.tolist() + ['missing']))
    educated_at_encoder = LabelEncoder().fit(educated_ats)

    all_genders = pd.concat([
        edges['Q_node1_gender'].fillna('missing'),
        edges['Q_node2_gender'].fillna('missing')
    ]).unique()
    genders = sorted(set(all_genders.tolist() + ['missing']))
    gender_encoder = LabelEncoder().fit(genders)

    all_countries = pd.concat([
        edges['Q_node1_country'].fillna('missing'),
        edges['Q_node2_country'].fillna('missing')
    ]).unique()
    countries = sorted(set(all_countries.tolist() + ['missing']))
    country_encoder = LabelEncoder().fit(countries)

    # 7. 划分训练集和测试集（基于节点）
    test_nodes = set(np.random.choice(remaining_nodes, size=int(len(remaining_nodes) * 0.2), replace=False))
    train_nodes = set(remaining_nodes) - test_nodes

    # 8. 创建训练图（将测试节点的职业标记为unknown）
    edges_train = edges.copy()
    # 避免泄露：只将测试节点的职业信息在训练图中置为unknown
    # 其他特征（work_location、educated_at、gender、country）保留，用于辅助预测职业
    for lvl in [1, 2, 3]:
        edges_train.loc[edges_train['Q_node1'].isin(test_nodes), f'Q_node1_level{lvl}_main_occ'] = "unknown"
        edges_train.loc[edges_train['Q_node2'].isin(test_nodes), f'Q_node2_level{lvl}_main_occ'] = "unknown"


    class OccupationDataset(Dataset):
        def __init__(self, nodes, edges_df, level=1, batch_size=512):
            print(f"Initializing dataset with {len(nodes)} nodes for level {level}")
            self.nodes = list(nodes)
            self.batch_size = batch_size
            self.level = level

            try:
                print("Building edge index and type...")
                # 构建完整图的edge_index和edge_type（只构建一次）
                edge_index_array = np.vstack([
                    edges_df['Q_node1_encoded'].values,
                    edges_df['Q_node2_encoded'].values
                ])
                self.edge_index = torch.from_numpy(edge_index_array).long()
                self.edge_type = torch.from_numpy(edges_df['r_encoded'].values).long()

                print("Moving tensors to device...")
                # 分批移动到GPU
                self.edge_index = self.edge_index.to(device)
                self.edge_type = self.edge_type.to(device)

                print("Initializing node features...")
                # 构建节点特征容器：occupation、work_location、educated_at、gender、country
                self.feature_tensors = {
                    'occupation': torch.zeros(len(node_encoder.classes_), dtype=torch.long),
                    'work_location': torch.zeros(len(node_encoder.classes_), dtype=torch.long),
                    'educated_at': torch.zeros(len(node_encoder.classes_), dtype=torch.long),
                    'gender': torch.zeros(len(node_encoder.classes_), dtype=torch.long),
                    'country': torch.zeros(len(node_encoder.classes_), dtype=torch.long),
                }

                print("Processing node features...")
                # 创建节点特征映射，确保每个节点只被填充一次
                node_features = {}
                
                # 收集所有节点的特征
                for _, row in edges_df.iterrows():
                    node1_id = row['Q_node1_encoded']
                    node2_id = row['Q_node2_encoded']
                    
                    # 为每个节点收集特征（如果还没有收集过）
                    if node1_id not in node_features:
                        node_features[node1_id] = {
                            'occupation': row[f'Q_node1_level{level}_main_occ'] if pd.notna(row[f'Q_node1_level{level}_main_occ']) else 'unknown',
                            'work_location': row['Q_node1_work_location'] if pd.notna(row['Q_node1_work_location']) else 'missing',
                            'educated_at': row['Q_node1_educated_at'] if pd.notna(row['Q_node1_educated_at']) else 'missing',
                            'gender': row['Q_node1_gender'] if pd.notna(row['Q_node1_gender']) else 'missing',
                            'country': row['Q_node1_country'] if pd.notna(row['Q_node1_country']) else 'missing'
                        }
                    
                    if node2_id not in node_features:
                        node_features[node2_id] = {
                            'occupation': row[f'Q_node2_level{level}_main_occ'] if pd.notna(row[f'Q_node2_level{level}_main_occ']) else 'unknown',
                            'work_location': row['Q_node2_work_location'] if pd.notna(row['Q_node2_work_location']) else 'missing',
                            'educated_at': row['Q_node2_educated_at'] if pd.notna(row['Q_node2_educated_at']) else 'missing',
                            'gender': row['Q_node2_gender'] if pd.notna(row['Q_node2_gender']) else 'missing',
                            'country': row['Q_node2_country'] if pd.notna(row['Q_node2_country']) else 'missing'
                        }
                
                # 填充特征张量
                for node_id, features in node_features.items():
                    self.feature_tensors['occupation'][node_id] = occ_encoders[level].transform([features['occupation']])[0]
                    self.feature_tensors['work_location'][node_id] = work_location_encoder.transform([features['work_location']])[0]
                    self.feature_tensors['educated_at'][node_id] = educated_at_encoder.transform([features['educated_at']])[0]
                    self.feature_tensors['gender'][node_id] = gender_encoder.transform([features['gender']])[0]
                    self.feature_tensors['country'][node_id] = country_encoder.transform([features['country']])[0]
                
                print(f"Processed {len(node_features)} unique nodes...")

                print("Moving node features to device...")
                for k in self.feature_tensors:
                    self.feature_tensors[k] = self.feature_tensors[k].to(device)

                print("Computing true labels...")
                # 预计算真实标签
                self.true_labels = {}
                for node in self.nodes:
                    node_edges = edges[
                        (edges['Q_node1'] == node) |
                        (edges['Q_node2'] == node)
                        ]
                    if len(node_edges) > 0:  # 确保找到了边
                        if node in node_edges['Q_node1'].values:
                            raw = node_edges.iloc[0][f'Q_node1_level{level}_main_occ']
                        else:
                            raw = node_edges.iloc[0][f'Q_node2_level{level}_main_occ']
                        raw = raw if pd.notna(raw) else 'unknown'
                        label = occ_encoders[level].transform([raw])[0]
                        self.true_labels[node] = label
                    else:
                        print(f"Warning: No edges found for node {node}")
                        # 使用unknown作为默认标签
                        self.true_labels[node] = occ_encoders[level].transform(['unknown'])[0]

                print("Dataset initialization completed successfully")

                if torch.cuda.is_available():
                    print(
                        f"GPU memory after dataset initialization: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
                    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

            except Exception as e:
                print(f"Error during dataset initialization: {str(e)}")
                if torch.cuda.is_available():
                    print(f"GPU memory at error: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
                    print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
                    torch.cuda.empty_cache()
                raise e

        def __len__(self):
            return batches_per_epoch  # 每个epoch只返回50个batch

        def __getitem__(self, idx):
            try:
                # 每次随机选择512个节点
                batch_nodes = np.random.choice(self.nodes, size=512, replace=False)

                # 复制所有特征张量，只mask职业特征
                masked_indices = [node_encoder.transform([node])[0] for node in batch_nodes]
                features = {}
                # occupation - 需要mask为unknown（预测目标）
                occ_tensor = self.feature_tensors['occupation'].clone()
                occ_tensor[masked_indices] = occ_encoders[self.level].transform(['unknown'])[0]
                features['occupation'] = occ_tensor
                # 其他特征 - 保留原始值（包括missing和真实值）
                features['work_location'] = self.feature_tensors['work_location'].clone()
                features['educated_at'] = self.feature_tensors['educated_at'].clone()
                features['gender'] = self.feature_tensors['gender'].clone()
                features['country'] = self.feature_tensors['country'].clone()

                # 获取真实标签
                labels = torch.tensor([self.true_labels[node] for node in batch_nodes],
                                      dtype=torch.long).to(device)

                return {
                    'edge_index': self.edge_index,
                    'edge_type': self.edge_type,
                    'features': features,
                    'masked_indices': torch.tensor(masked_indices, dtype=torch.long).to(device),
                    'labels': labels
                }
            except Exception as e:
                print(f"Error in __getitem__ for batch {idx}: {str(e)}")
                raise e


    class RGCNModel(nn.Module):
        def __init__(self, num_relations, hidden_dim, feature_num_classes: dict):
            super().__init__()
            # 为每个特征创建Embedding
            self.embeddings = nn.ModuleDict({
                feature_name: nn.Embedding(num_classes, hidden_dim)
                for feature_name, num_classes in feature_num_classes.items()
            })
            in_channels = hidden_dim * len(self.embeddings)
            self.conv1 = RGCNConv(in_channels, hidden_dim, num_relations, num_bases=30)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=30)
            # 预测occupation类别数量
            self.classifier = nn.Linear(hidden_dim, feature_num_classes['occupation'])

        def forward(self, features_dict, edge_index, edge_type):
            # 获取并拼接所有启用特征的embedding
            embeddings_list = []
            for feature_name, emb_layer in self.embeddings.items():
                embeddings_list.append(emb_layer(features_dict[feature_name]))
            combined_features = torch.cat(embeddings_list, dim=1)

            # 应用GNN层
            h = self.conv1(combined_features, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)
            h = self.conv2(h, edge_index, edge_type)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)

            # 返回预测结果
            return self.classifier(h)


    def train_epoch(model, loader, optimizer, device, selected_features):
        model.train()
        total_loss = 0

        for batch in loader:
            optimizer.zero_grad()

            # 获取数据
            edge_index = batch['edge_index'][0]
            edge_type = batch['edge_type'][0]
            # 只选择当前实验组指定的特征
            features_all = {k: v[0] for k, v in batch['features'].items()}
            features_selected = {k: features_all[k] for k in selected_features}
            masked_indices = batch['masked_indices'][0]
            labels = batch['labels'][0]

            # 前向传播
            out = model(features_selected, edge_index, edge_type)
            pred = out[masked_indices]

            # 计算损失
            loss = F.cross_entropy(pred, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


    @torch.no_grad()
    def evaluate(model, loader, device, selected_features, phase="val"):
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        for batch in loader:
            edge_index = batch['edge_index'][0]
            edge_type = batch['edge_type'][0]
            # 只选择当前实验组指定的特征
            features_all = {k: v[0] for k, v in batch['features'].items()}
            features_selected = {k: features_all[k] for k in selected_features}
            masked_indices = batch['masked_indices'][0]
            labels = batch['labels'][0]
            out = model(features_selected, edge_index, edge_type)
            pred = out[masked_indices]
            loss = F.cross_entropy(pred, labels)
            total_loss += loss.item()
            pred_classes = pred.argmax(dim=1)
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        avg_loss = total_loss / len(loader)
        results = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        # 只在test阶段输出详细指标
        if phase == "test":
            pass  # 论文用的输出已在主循环中统一输出
        return results


    # 定义实验分组（特征组合）
    # 每个实验组只使用指定的特征组合，预测目标始终是occupation
    experiment_feature_sets = [
        ['occupation'],                                    # 基线：只用职业信息
        ['occupation', 'work_location'],                   # 职业 + 工作地点
        ['occupation', 'educated_at'],                     # 职业 + 教育背景
        ['occupation', 'gender'],                          # 职业 + 性别
        ['occupation', 'country'],                         # 职业 + 国家
        ['occupation', 'work_location', 'educated_at'],    # 职业 + 工作地点 + 教育背景
        ['occupation', 'country', 'educated_at'],          # 职业 + 国家 + 教育背景
        ['occupation', 'country', 'gender'],               # 职业 + 国家 + 性别
        ['occupation', 'country', 'gender', 'educated_at'], # 职业 + 国家 + 性别 + 教育背景
    ]

    # 对每个level与特征组合进行训练
    # for level in [1, 2, 3]:
    for level in [3]:
        print(f"\n开始训练 Level {level} 模型（多特征对比）...")

        try:
            # 创建数据集
            train_dataset = OccupationDataset(train_nodes, edges_train, level=level)
            test_dataset = OccupationDataset(test_nodes, edges_train, level=level)

            # DataLoader
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            # 编码器类别数
            occ_num_classes = len(occ_encoders[level].classes_)
            work_loc_num_classes = len(work_location_encoder.classes_)
            edu_num_classes = len(educated_at_encoder.classes_)
            gender_num_classes = len(gender_encoder.classes_)
            country_num_classes = len(country_encoder.classes_)

            for feature_set in experiment_feature_sets:
                set_name = "+".join(feature_set)
                print(f"\n[Level {level}] 开始实验：{set_name}")
                print(f"使用特征：{feature_set}")
                print(f"预测目标：occupation")

                # 为当前特征组合构造num_classes字典
                feature_num_classes = {}
                for f in feature_set:
                    if f == 'occupation':
                        feature_num_classes[f] = occ_num_classes
                    elif f == 'work_location':
                        feature_num_classes[f] = work_loc_num_classes
                    elif f == 'educated_at':
                        feature_num_classes[f] = edu_num_classes
                    elif f == 'gender':
                        feature_num_classes[f] = gender_num_classes
                    elif f == 'country':
                        feature_num_classes[f] = country_num_classes

                # 创建模型
                model = RGCNModel(num_relations, hidden_dim=64, feature_num_classes=feature_num_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

                # 训练和评估
                best_val_f1 = 0
                best_model = None

                for epoch in range(num_epochs):
                    try:
                        # 训练
                        loss = train_epoch(model, train_loader, optimizer, device, feature_set)

                        # 评估
                        train_results = evaluate(model, train_loader, device, feature_set, "train")
                        val_results = evaluate(model, test_loader, device, feature_set, "val")

                        # 保存最佳模型
                        if val_results['f1'] > best_val_f1:
                            best_val_f1 = val_results['f1']
                            best_model = copy.deepcopy(model)

                        # 精简输出
                        print(
                            f"[Level {level}][{set_name}][Epoch {epoch + 1}/{num_epochs}] Train Loss: {loss:.4f} | Train Acc: {train_results['accuracy']:.4f} | Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy']:.4f} | Val F1: {val_results['f1']:.4f}")

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if best_model is not None:
                                model = copy.deepcopy(best_model)
                                continue
                            else:
                                break
                        else:
                            raise e

                # 最终测试
                if best_model is not None:
                    test_results = evaluate(best_model, test_loader, device, feature_set, "test")
                    print(f"\n[Level {level}][{set_name}] Test Results:")
                    print(f"Accuracy: {test_results['accuracy']:.4f}")
                    print(f"Precision: {test_results['precision']:.4f}")
                    print(f"Recall: {test_results['recall']:.4f}")
                    print(f"F1 Score: {test_results['f1']:.4f}")

                # 释放显存
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except RuntimeError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

except Exception as e:
    print(f"Error: {str(e)}")
    if torch.cuda.is_available():
        print(f"Final GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Final GPU memory cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
        torch.cuda.empty_cache()
    raise e
