import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import device, BATCH_SIZE, BATCHES_PER_EPOCH


class OccupationDataset(Dataset):
    def __init__(self, nodes, edges_df, level, encoders, node_encoder):
        self.nodes = list(nodes)
        self.level = level
        self.encoders = encoders
        self.node_encoder = node_encoder
        
        self._build_graph_tensors(edges_df)
        self._build_node_features(edges_df)
        self._compute_true_labels(edges_df)
    
    def _build_graph_tensors(self, edges_df):
        """Build edge index and edge type tensors"""
        edge_index_array = np.vstack([
            edges_df['Q_node1_encoded'].values,
            edges_df['Q_node2_encoded'].values
        ])
        self.edge_index = torch.from_numpy(edge_index_array).long().to(device)
        self.edge_type = torch.from_numpy(edges_df['r_encoded'].values).long().to(device)
    
    def _build_node_features(self, edges_df):
        """Build node feature tensors"""
        self.feature_tensors = {
            'occupation': torch.zeros(len(self.node_encoder.classes_), dtype=torch.long),
            'work_location': torch.zeros(len(self.node_encoder.classes_), dtype=torch.long),
            'educated_at': torch.zeros(len(self.node_encoder.classes_), dtype=torch.long),
            'gender': torch.zeros(len(self.node_encoder.classes_), dtype=torch.long),
            'country': torch.zeros(len(self.node_encoder.classes_), dtype=torch.long),
        }
        
        node_features = {}
        
        for _, row in edges_df.iterrows():
            node1_id = row['Q_node1_encoded']
            node2_id = row['Q_node2_encoded']
            
            if node1_id not in node_features:
                node_features[node1_id] = {
                    'occupation': row[f'Q_node1_level{self.level}_main_occ'] if pd.notna(row[f'Q_node1_level{self.level}_main_occ']) else 'unknown',
                    'work_location': row['Q_node1_work_location'] if pd.notna(row['Q_node1_work_location']) else 'missing',
                    'educated_at': row['Q_node1_educated_at'] if pd.notna(row['Q_node1_educated_at']) else 'missing',
                    'gender': row['Q_node1_gender'] if pd.notna(row['Q_node1_gender']) else 'missing',
                    'country': row['Q_node1_country'] if pd.notna(row['Q_node1_country']) else 'missing'
                }
            
            if node2_id not in node_features:
                node_features[node2_id] = {
                    'occupation': row[f'Q_node2_level{self.level}_main_occ'] if pd.notna(row[f'Q_node2_level{self.level}_main_occ']) else 'unknown',
                    'work_location': row['Q_node2_work_location'] if pd.notna(row['Q_node2_work_location']) else 'missing',
                    'educated_at': row['Q_node2_educated_at'] if pd.notna(row['Q_node2_educated_at']) else 'missing',
                    'gender': row['Q_node2_gender'] if pd.notna(row['Q_node2_gender']) else 'missing',
                    'country': row['Q_node2_country'] if pd.notna(row['Q_node2_country']) else 'missing'
                }
        
        for node_id, features in node_features.items():
            self.feature_tensors['occupation'][node_id] = self.encoders['occ_encoders'][self.level].transform([features['occupation']])[0]
            self.feature_tensors['work_location'][node_id] = self.encoders['work_location_encoder'].transform([features['work_location']])[0]
            self.feature_tensors['educated_at'][node_id] = self.encoders['educated_at_encoder'].transform([features['educated_at']])[0]
            self.feature_tensors['gender'][node_id] = self.encoders['gender_encoder'].transform([features['gender']])[0]
            self.feature_tensors['country'][node_id] = self.encoders['country_encoder'].transform([features['country']])[0]
        
        for k in self.feature_tensors:
            self.feature_tensors[k] = self.feature_tensors[k].to(device)
    
    def _compute_true_labels(self, edges_df):
        """Compute true labels for nodes"""
        self.true_labels = {}
        for node in self.nodes:
            node_edges = edges_df[
                (edges_df['Q_node1'] == node) |
                (edges_df['Q_node2'] == node)
            ]
            if len(node_edges) > 0:
                if node in node_edges['Q_node1'].values:
                    raw = node_edges.iloc[0][f'Q_node1_level{self.level}_main_occ']
                else:
                    raw = node_edges.iloc[0][f'Q_node2_level{self.level}_main_occ']
                raw = raw if pd.notna(raw) else 'unknown'
                label = self.encoders['occ_encoders'][self.level].transform([raw])[0]
                self.true_labels[node] = label
            else:
                self.true_labels[node] = self.encoders['occ_encoders'][self.level].transform(['unknown'])[0]
    
    def __len__(self):
        return BATCHES_PER_EPOCH
    
    def __getitem__(self, idx):
        batch_nodes = np.random.choice(self.nodes, size=BATCH_SIZE, replace=False)
        
        masked_indices = [self.node_encoder.transform([node])[0] for node in batch_nodes]
        features = {}
        
        occ_tensor = self.feature_tensors['occupation'].clone()
        occ_tensor[masked_indices] = self.encoders['occ_encoders'][self.level].transform(['unknown'])[0]
        features['occupation'] = occ_tensor
        
        features['work_location'] = self.feature_tensors['work_location'].clone()
        features['educated_at'] = self.feature_tensors['educated_at'].clone()
        features['gender'] = self.feature_tensors['gender'].clone()
        features['country'] = self.feature_tensors['country'].clone()
        
        labels = torch.tensor([self.true_labels[node] for node in batch_nodes], dtype=torch.long).to(device)
        
        return {
            'edge_index': self.edge_index,
            'edge_type': self.edge_type,
            'features': features,
            'masked_indices': torch.tensor(masked_indices, dtype=torch.long).to(device),
            'labels': labels
        }
