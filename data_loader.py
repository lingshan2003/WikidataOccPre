import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import device, TEST_SPLIT_RATIO, EDGES_FILE, ATTRIBUTES_FILE


class DataLoader:
    def __init__(self):
        self.edges = None
        self.attributes = None
        self.node_encoder = None
        self.relation_encoder = None
        self.occ_encoders = {}
        self.work_location_encoder = None
        self.educated_at_encoder = None
        self.gender_encoder = None
        self.country_encoder = None
        self.train_nodes = None
        self.test_nodes = None
        self.edges_train = None
        
    def load_data(self):
        """Load and preprocess the data"""
        self.edges = pd.read_csv(EDGES_FILE, sep=' ', header=None, names=['Q_node1', 'r', 'Q_node2'])
        self.attributes = pd.read_csv(ATTRIBUTES_FILE, sep='\t')
        
        self._add_features()
        self._remove_missing_occupations()
        self._create_encoders()
        self._split_train_test()
        self._prepare_training_data()
        
    def _add_features(self):
        """Add features to edges dataframe"""
        # Occupation features
        for level in [1, 2, 3]:
            self.edges[f'Q_node1_level{level}_main_occ'] = self.edges['Q_node1'].map(
                self.attributes.set_index('Q_node')[f'level{level}_main_occ']
            )
            self.edges[f'Q_node2_level{level}_main_occ'] = self.edges['Q_node2'].map(
                self.attributes.set_index('Q_node')[f'level{level}_main_occ']
            )
        
        # Other features
        feature_mappings = {
            'work_location': 'work location',
            'educated_at': 'educated at',
            'gender': 'gender',
            'country': 'country'
        }
        
        for feature, attr_name in feature_mappings.items():
            self.edges[f'Q_node1_{feature}'] = self.edges['Q_node1'].map(
                self.attributes.set_index('Q_node')[attr_name]
            )
            self.edges[f'Q_node2_{feature}'] = self.edges['Q_node2'].map(
                self.attributes.set_index('Q_node')[attr_name]
            )
    
    def _remove_missing_occupations(self):
        """Remove edges with missing occupation information"""
        initial_count = len(self.edges)
        self.edges = self.edges.dropna(subset=['Q_node1_level1_main_occ', 'Q_node2_level1_main_occ'])
        print(f"Removed {initial_count - len(self.edges)} edges with missing occupation data")
    
    def _create_encoders(self):
        """Create encoders for nodes, relations, and features"""
        remaining_nodes = pd.concat([self.edges['Q_node1'], self.edges['Q_node2']]).unique()
        remaining_relations = self.edges['r'].unique()
        
        self.node_encoder = LabelEncoder().fit(remaining_nodes)
        self.relation_encoder = LabelEncoder().fit(remaining_relations)
        
        # Encode edges
        self.edges['Q_node1_encoded'] = self.node_encoder.transform(self.edges['Q_node1'])
        self.edges['Q_node2_encoded'] = self.node_encoder.transform(self.edges['Q_node2'])
        self.edges['r_encoded'] = self.relation_encoder.transform(self.edges['r'])
        
        # Create occupation encoders
        for level in [1, 2, 3]:
            level_occupations = sorted(set(
                pd.concat([
                    self.edges[f'Q_node1_level{level}_main_occ'],
                    self.edges[f'Q_node2_level{level}_main_occ']
                ]).unique().tolist() + ["unknown"]
            ))
            self.occ_encoders[level] = LabelEncoder().fit(level_occupations)
        
        # Create other feature encoders
        self._create_feature_encoders()
    
    def _create_feature_encoders(self):
        """Create encoders for non-occupation features"""
        feature_configs = {
            'work_location': 'work_location',
            'educated_at': 'educated_at',
            'gender': 'gender',
            'country': 'country'
        }
        
        for feature_name, col_prefix in feature_configs.items():
            all_values = pd.concat([
                self.edges[f'Q_node1_{col_prefix}'].fillna('missing'),
                self.edges[f'Q_node2_{col_prefix}'].fillna('missing')
            ]).unique()
            values = sorted(set(all_values.tolist() + ['missing']))
            
            if feature_name == 'work_location':
                self.work_location_encoder = LabelEncoder().fit(values)
            elif feature_name == 'educated_at':
                self.educated_at_encoder = LabelEncoder().fit(values)
            elif feature_name == 'gender':
                self.gender_encoder = LabelEncoder().fit(values)
            elif feature_name == 'country':
                self.country_encoder = LabelEncoder().fit(values)
    
    def _split_train_test(self):
        """Split nodes into train and test sets"""
        remaining_nodes = pd.concat([self.edges['Q_node1'], self.edges['Q_node2']]).unique()
        self.test_nodes = set(np.random.choice(remaining_nodes, size=int(len(remaining_nodes) * TEST_SPLIT_RATIO), replace=False))
        self.train_nodes = set(remaining_nodes) - self.test_nodes
    
    def _prepare_training_data(self):
        """Prepare training data by masking test node occupations"""
        self.edges_train = self.edges.copy()
        
        for level in [1, 2, 3]:
            self.edges_train.loc[self.edges_train['Q_node1'].isin(self.test_nodes), f'Q_node1_level{level}_main_occ'] = "unknown"
            self.edges_train.loc[self.edges_train['Q_node2'].isin(self.test_nodes), f'Q_node2_level{level}_main_occ'] = "unknown"
    
    def get_encoder_info(self):
        """Get information about encoders"""
        return {
            'num_nodes': len(self.node_encoder.classes_),
            'num_relations': len(self.relation_encoder.classes_),
            'occ_encoders': self.occ_encoders,
            'work_location_encoder': self.work_location_encoder,
            'educated_at_encoder': self.educated_at_encoder,
            'gender_encoder': self.gender_encoder,
            'country_encoder': self.country_encoder
        }
