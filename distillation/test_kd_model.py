import os
import torch
import numpy as np
from utils import *
from tqdm import tqdm
from models.huggingface_clip import HFLoRACLIPVisionModel
from peft import LoraConfig

"""
def test_clip_model(model_path, datasets=["mnist", "svhn"]):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model configuration
    config_name = 'multidataset_hf_clip_kd'
    raw_config = get_config_from_name(config_name, device=device)
    
    # Configure LoRA for model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    # Initialize the model
    model = HFLoRACLIPVisionModel(
        model_name="openai/clip-vit-base-patch32",
        cache_dir='/nethome/becsedi3/flash/logs/kd_distill/cache',
        lora_config=lora_config.__dict__,
        device=device
    )
    
    # Load the fine-tuned weights
    print(f"Loading model from: {model_path}")
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = model.train_preprocess
        dataset_config['eval_preprocess'] = model.val_preprocess
    
    # Load data
    data_loaders = prepare_data(raw_config['dataset'], device=device)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    
    # Load class vectors for each dataset
    clip_encodings = []
    for i in raw_config['dataset']:
        clip_encodings.append(get_clip_encodings(i['student_clip_encodings']))
    
    # Evaluate on each requested dataset
    results = {}
    for dataset_name in datasets:
        if dataset_name not in dataset_names:
            print(f"Dataset {dataset_name} not found in configuration!")
            continue
        
        idx = np.where(dataset_names == dataset_name)[0][0]
        loader_dict = data_loaders[idx]
        class_vectors = clip_encodings[idx]
        
        print(f"\nEvaluating on {dataset_name}...")
        
        # Make sure we have a test loader
        if 'test' not in loader_dict['test']:
            print(f"Warning: 'test' key not found in loader_dict['test'] for {dataset_name}. Using available keys.")
            test_loader = next(iter(loader_dict['test'].values()))
        else:
            test_loader = loader_dict['test']['test']
        
        test_acc, test_loss = evaluate_cliphead(
            model, 
            test_loader, 
            class_vectors=class_vectors, 
            return_loss=True
        )
        
        print(f"{dataset_name.upper()} Test Accuracy: {test_acc:.4f}")
        print(f"{dataset_name.upper()} Test Loss: {test_loss:.4f}")
        
        results[dataset_name] = {
            'accuracy': test_acc,
            'loss': test_loss
        }
    
    # Print summary of results
    print("\n======= EVALUATION SUMMARY =======")
    for dataset, metrics in results.items():
        print(f"{dataset.upper()}: Accuracy = {metrics['accuracy']:.4f}, Loss = {metrics['loss']:.4f}")
    
    return results

"""

def test_clip_model(model_path, datasets=["mnist", "svhn"]):
    """
    Load a fine-tuned CLIP-ViT-B/32 model and evaluate it on MNIST and SVHN test sets.
    
    Args:
        model_path: Path to the saved fine-tuned model
        datasets: List of datasets to evaluate on
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model configuration
    config_name = 'multidataset_hf_clip_kd'
    raw_config = get_config_from_name(config_name, device=device)
    
    # Configure LoRA for model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    # Initialize the model
    model = HFLoRACLIPVisionModel(
        model_name="openai/clip-vit-base-patch32",
        cache_dir='/nethome/becsedi3/flash/logs/kd_distill/cache',
        lora_config=lora_config.__dict__,
        device=device
    )
    
    # Load the fine-tuned weights
    print(f"Loading model from: {model_path}")
    model_state_dict = torch.load(model_path, map_location=device)
    
    # Fix the state dict key mismatch by remapping keys
    # The saved model uses 'lora_model' prefix but our model expects 'vision_model'
    new_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith('lora_model'):
            # Replace 'lora_model' with 'vision_model'
            new_key = key.replace('lora_model', 'vision_model')
            new_state_dict[new_key] = value
        else:
            # Keep any other keys as is
            new_state_dict[key] = value
    
    # Load the remapped state dict
    try:
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully with key remapping.")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        
        # If the above fails, try loading without strict matching
        # This will load parameters that match and ignore those that don't
        print("Attempting to load with strict=False...")
        incompatible = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded model with {len(incompatible.missing_keys)} missing keys and {len(incompatible.unexpected_keys)} unexpected keys.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Continue with the rest of your code...
    # [existing code for dataset preprocessing and evaluation]
    
    # Add preprocessing functions to dataset config
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = model.train_preprocess
        dataset_config['eval_preprocess'] = model.val_preprocess
    
    # Load data
    data_loaders = prepare_data(raw_config['dataset'], device=device)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    
    # Load class vectors for each dataset
    clip_encodings = []
    for i in raw_config['dataset']:
        clip_encodings.append(get_clip_encodings(i['student_clip_encodings']))
    
    # Evaluate on each requested dataset
    results = {}
    for dataset_name in datasets:
        if dataset_name not in dataset_names:
            print(f"Dataset {dataset_name} not found in configuration!")
            continue
        
        idx = np.where(dataset_names == dataset_name)[0][0]
        loader_dict = data_loaders[idx]
        class_vectors = clip_encodings[idx]
        
        print(f"\nEvaluating on {dataset_name}...")
        
        # Make sure we have a test loader
        if 'test' not in loader_dict['test']:
            print(f"Warning: 'test' key not found in loader_dict['test'] for {dataset_name}. Using available keys.")
            test_loader = next(iter(loader_dict['test'].values()))
        else:
            test_loader = loader_dict['test']['test']
        
        test_acc, test_loss = evaluate_cliphead(
            model, 
            test_loader, 
            class_vectors=class_vectors, 
            return_loss=True
        )
        
        print(f"{dataset_name.upper()} Test Accuracy: {test_acc:.4f}")
        print(f"{dataset_name.upper()} Test Loss: {test_loss:.4f}")
        
        results[dataset_name] = {
            'accuracy': test_acc,
            'loss': test_loss
        }
    
    # Print summary of results
    print("\n======= EVALUATION SUMMARY =======")
    for dataset, metrics in results.items():
        print(f"{dataset.upper()}: Accuracy = {metrics['accuracy']:.4f}, Loss = {metrics['loss']:.4f}")
    
    return results

if __name__ == "__main__":
    distilled_model_path = '/nethome/becsedi3/flash/model-checkpoints/kd_lora_multidataset/best_models/KD_svhn_from_mnist_lr_0.0005_epochs_5_wd_5e-05_alpha_0.25_temp_1.0_rank16.pt'
    ft_model_path = '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/svhn_lora-r16-alpha16-dropout0.1.pt'
    
    # Evaluate on MNIST and SVHN
    print("Fine-tuned results: ")
    ft_results = test_clip_model(ft_model_path)
    
    print("Distilled results: ")
    distilled_results = test_clip_model(distilled_model_path)
    
    
    