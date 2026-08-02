import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from config import HIDDEN_DIM, NUM_BASES, DROPOUT_RATE


class RGCNModel(nn.Module):
    def __init__(self, num_relations, hidden_dim, feature_num_classes):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            feature_name: nn.Embedding(num_classes, hidden_dim)
            for feature_name, num_classes in feature_num_classes.items()
        })
        
        in_channels = hidden_dim * len(self.embeddings)
        self.conv1 = RGCNConv(in_channels, hidden_dim, num_relations, num_bases=NUM_BASES)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations, num_bases=NUM_BASES)
        self.classifier = nn.Linear(hidden_dim, feature_num_classes['occupation'])

    def forward(self, features_dict, edge_index, edge_type):
        embeddings_list = []
        for feature_name, emb_layer in self.embeddings.items():
            embeddings_list.append(emb_layer(features_dict[feature_name]))
        combined_features = torch.cat(embeddings_list, dim=1)

        h = self.conv1(combined_features, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT_RATE, training=self.training)
        h = self.conv2(h, edge_index, edge_type)
        h = F.relu(h)
        h = F.dropout(h, p=DROPOUT_RATE, training=self.training)

        return self.classifier(h)
