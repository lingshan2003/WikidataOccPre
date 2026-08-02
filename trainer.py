import torch
import torch.nn.functional as F
import numpy as np
import copy
from sklearn.metrics import precision_recall_fscore_support
from config import device, LEARNING_RATE, WEIGHT_DECAY


def train_epoch(model, loader, optimizer, selected_features):
    model.train()
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()

        edge_index = batch['edge_index'][0]
        edge_type = batch['edge_type'][0]
        features_all = {k: v[0] for k, v in batch['features'].items()}
        features_selected = {k: features_all[k] for k in selected_features}
        masked_indices = batch['masked_indices'][0]
        labels = batch['labels'][0]

        out = model(features_selected, edge_index, edge_type)
        pred = out[masked_indices]

        loss = F.cross_entropy(pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, selected_features, phase="val"):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    for batch in loader:
        edge_index = batch['edge_index'][0]
        edge_type = batch['edge_type'][0]
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
    
    return results


def train_model(model, train_loader, test_loader, selected_features, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    best_val_f1 = 0
    best_model = None

    for epoch in range(num_epochs):
        try:
            loss = train_epoch(model, train_loader, optimizer, selected_features)
            train_results = evaluate(model, train_loader, selected_features, "train")
            val_results = evaluate(model, test_loader, selected_features, "val")

            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                best_model = copy.deepcopy(model)

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

    return best_model, best_val_f1
