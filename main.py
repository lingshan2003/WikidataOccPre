import torch
from torch.utils.data import DataLoader
from data_loader import DataLoader as GraphDataLoader
from dataset import OccupationDataset
from model import RGCNModel
from trainer import train_model, evaluate
from utils import setup_device, clear_gpu_memory, get_feature_num_classes, print_experiment_info
from config import NUM_EPOCHS, HIDDEN_DIM, EXPERIMENT_FEATURE_SETS


def main():
    device, device_name = setup_device()
    print(f"Using device: {device_name}")
    
    try:
        # Load data
        data_loader = GraphDataLoader()
        data_loader.load_data()
        encoder_info = data_loader.get_encoder_info()
        
        # Train for level 3 only (as in original code)
        for level in [3]:
            print(f"\nTraining Level {level} model...")
            
            # Create datasets
            train_dataset = OccupationDataset(
                data_loader.train_nodes, 
                data_loader.edges_train, 
                level, 
                encoder_info,
                data_loader.node_encoder
            )
            test_dataset = OccupationDataset(
                data_loader.test_nodes, 
                data_loader.edges_train, 
                level, 
                encoder_info,
                data_loader.node_encoder
            )
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            # Run experiments for each feature set
            for feature_set in EXPERIMENT_FEATURE_SETS:
                print_experiment_info(level, feature_set)
                
                # Get feature dimensions
                all_feature_classes = get_feature_num_classes(encoder_info, level)
                feature_num_classes = {f: all_feature_classes[f] for f in feature_set}
                
                # Create model
                model = RGCNModel(
                    encoder_info['num_relations'], 
                    HIDDEN_DIM, 
                    feature_num_classes
                ).to(device)
                
                # Train model
                best_model, best_val_f1 = train_model(
                    model, train_loader, test_loader, feature_set, NUM_EPOCHS
                )
                
                # Final evaluation
                if best_model is not None:
                    test_results = evaluate(best_model, test_loader, feature_set, "test")
                    print(f"Test Results - Accuracy: {test_results['accuracy']:.4f}, "
                          f"F1: {test_results['f1']:.4f}")
                
                # Clean up
                del model
                clear_gpu_memory()
                
    except Exception as e:
        print(f"Error: {str(e)}")
        clear_gpu_memory()
        raise e


if __name__ == "__main__":
    main()
