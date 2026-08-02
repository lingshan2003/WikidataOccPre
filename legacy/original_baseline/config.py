import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
NUM_EPOCHS = 5
BATCH_SIZE = 512
HIDDEN_DIM = 64
PRINT_EVERY = 1
BATCHES_PER_EPOCH = 100

# Model parameters
NUM_BASES = 30
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.5

# Data parameters
TEST_SPLIT_RATIO = 0.2

# Feature sets for experiments
EXPERIMENT_FEATURE_SETS = [
    ['occupation'],
    ['occupation', 'work_location'],
    ['occupation', 'educated_at'],
    ['occupation', 'gender'],
    ['occupation', 'country'],
    ['occupation', 'work_location', 'educated_at'],
    ['occupation', 'country', 'educated_at'],
    ['occupation', 'country', 'gender'],
    ['occupation', 'country', 'gender', 'educated_at'],
]

# File paths
EDGES_FILE = 'Q_R_Q.txt'
ATTRIBUTES_FILE = 'filtered_Q_attribute(60.8w).txt'
