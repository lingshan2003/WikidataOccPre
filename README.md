# RGCN for Occupation Prediction

This repository contains a Relational Graph Convolutional Network (RGCN) implementation for predicting occupations in knowledge graphs using multiple node features.

## 🚀 Features

- **Multi-feature Support**: Combines occupation, work location, education background, gender, country and other features
- **Modular Design**: Clean code structure, easy to maintain and extend
- **GPU Optimization**: Automatic GPU detection and memory management
- **Experiment Framework**: Systematic evaluation of different feature combinations

## 📁 Project Structure

```
├── config.py          # Configuration file
├── data_loader.py     # Data loading
├── dataset.py         # Dataset class
├── model.py           # RGCN model
├── extended_data.py   # Q_R_Q_extended.csv preparation utilities
├── RGAT_DESIGN.md     # relation-aware GAT experiment design
├── trainer.py         # Training functions
├── utils.py           # Utility functions
├── main.py            # Main program
├── test_modules.py    # Test script
└── requirements.txt   # Dependencies
```

## 🛠️ Installation

```bash
git clone <repository-url>
cd RGCN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On a CUDA server, install the PyTorch wheel matching the server's CUDA version
from the official PyTorch instructions before installing the remaining entries
in `requirements.txt`.

## 📊 Data Format

- `Q_R_Q.txt`: Edge data `Q_node1 relation Q_node2`
- `filtered_Q_attribute(60.8w).txt`: Node attributes (tab-separated)

## 🎯 Usage

```bash
# Run training
python main.py

# Test modules
python test_modules.py
```

The root `.venv` directory is intentionally ignored by Git. Recreate it on the
server instead of uploading the local macOS environment.

## ⚙️ Configuration

Modify parameters in `config.py`:
- `NUM_EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Batch size
- `HIDDEN_DIM`: Hidden layer dimension
- `LEARNING_RATE`: Learning rate

## 🧠 Model Architecture

1. **Feature Embedding Layer**: Multi-feature embeddings
2. **RGCN Layer**: Relational graph convolution
3. **Classifier**: Occupation prediction

## Next-generation experiment

The original `main.py` is a simple full-graph R-GCN baseline.  The new
`RelationalGATClassifier` in `model.py` and `extended_data.py` prepare a
relation-aware attention experiment over `Q_R_Q_extended.txt`.  It is designed
for future categorical, numeric, and pre-computed vector attributes.  See
[`RGAT_DESIGN.md`](RGAT_DESIGN.md) for the masking rule, reverse relations,
neighbor sampling, and attention explanation protocol.

## 📈 Results

Output metrics: Accuracy, Precision, Recall, F1-Score

## 📦 Dependencies

- torch >= 1.9.0
- torch-geometric >= 2.0.0
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
