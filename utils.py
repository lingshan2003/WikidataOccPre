import torch
from config import device


def setup_device():
    """Setup and return device information"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device('cuda'), torch.cuda.get_device_name(0)
    else:
        return torch.device('cpu'), "CPU"


def clear_gpu_memory():
    """Clear GPU memory if available"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_feature_num_classes(encoders, level):
    """Get number of classes for each feature"""
    return {
        'occupation': len(encoders['occ_encoders'][level].classes_),
        'work_location': len(encoders['work_location_encoder'].classes_),
        'educated_at': len(encoders['educated_at_encoder'].classes_),
        'gender': len(encoders['gender_encoder'].classes_),
        'country': len(encoders['country_encoder'].classes_)
    }


def print_experiment_info(level, feature_set):
    """Print experiment information"""
    set_name = "+".join(feature_set)
    print(f"\n[Level {level}] Experiment: {set_name}")
    print(f"Features: {feature_set}")
    print(f"Target: occupation")
