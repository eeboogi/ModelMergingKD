import os
import torch
import random
import numpy as np
from utils import *
from tqdm import tqdm
from copy import deepcopy
from peft import LoraConfig
from torch.nn.functional import kl_div
from models.huggingface_clip import HFLoRACLIPVisionModel
import wandb
from torch.cuda.amp import GradScaler, autocast

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            print(self.counter)
            if self.counter >= self.patience:
                return True
        return False

def train_cliphead_kd_lora(student_model, teacher_model, train_loader, val_loader, test_loader, 
                        class_vectors, remap_class_idxs=None, eval_class_vectors=None, 
                        clip_mapper=None, hyper_param_config=None, eval_freq=2):
    """Train a cliphead model with knowledge distillation.
    
    Args:
        student_model: student cliphead model (to be trained)
        teacher_model: teacher cliphead model (frozen)
        train_loader: dataloader to train on
        test_loader: dataloader to test on
        class_vectors: clip label encodings
        remap_class_idxs: array or mapping from true class labels to those expected given the task
        
    Returns:
        model: trained cliphead model
        test_acc: test accuracy
        test_loss: test loss
        val_acc: validation accuracy
        val_loss: validation loss
    """

    wandb.log(hyper_param_config)
    epochs = hyper_param_config['epochs']
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=hyper_param_config['lr'], weight_decay=hyper_param_config['wd'])
    ne_iters = len(train_loader)
    
    # Setup learning rate scheduler
    scheduler = cosine_lr(optimizer, hyper_param_config['lr'], hyper_param_config['warm_up'], epochs * ne_iters)
    
    scaler = GradScaler()
    ce_loss_fn = CrossEntropyLoss(label_smoothing=hyper_param_config['label_smoothing'])
    device = get_device(student_model)
    
    # Ensure teacher model is in eval mode and on the same device
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    pbar = tqdm(range(epochs), desc=f'KD finetuning')
    
    for epoch in pbar:
        student_model = student_model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            step = i + epoch * ne_iters
            
            optimizer.zero_grad(set_to_none=True)
            
            # Get student predictions
            with autocast():
                student_encodings = student_model(inputs.to(device))
                student_normed = student_encodings / student_encodings.norm(dim=-1, keepdim=True)
                student_logits = (100.0 * student_normed @ class_vectors.T)
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_encodings = teacher_model(inputs.to(device))
                    teacher_normed = teacher_encodings / teacher_encodings.norm(dim=-1, keepdim=True)
                    teacher_logits = (100.0 * teacher_normed @ class_vectors.T)
                
                if remap_class_idxs is not None:
                    remapped_labels = remap_class_idxs[labels].to(device)
                else:
                    remapped_labels = labels.to(device)
                
                # Calculate cross-entropy loss against true labels
                ce_loss = ce_loss_fn(student_logits, remapped_labels)
                
                # Calculate KL divergence loss
                temperature = hyper_param_config['temperature']
                soft_target = (teacher_logits / temperature).softmax(dim=-1)
                soft_pred = (student_logits / temperature).log_softmax(dim=-1)
                kl_loss = kl_div(soft_pred, soft_target, reduction='batchmean') * (temperature ** 2)
                
                # Combined loss
                kd_weight = hyper_param_config['kd_weight']
                loss = (1 - kd_weight) * ce_loss + kd_weight * kl_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler(step)
            
        # Evaluate on validation and test sets
        val_acc, val_loss = evaluate_cliphead(student_model, val_loader, class_vectors=class_vectors, 
                                            remap_class_idxs=remap_class_idxs, return_loss=True)
        test_acc, test_loss = evaluate_cliphead(student_model, test_loader, class_vectors=class_vectors, 
                                            remap_class_idxs=remap_class_idxs, return_loss=True)
        
        pbar.set_description(f'KD finetuning, val acc: {val_acc:.4f}')
        print(f'Epoch {epoch}, Test Acc: {test_acc}, Test Loss: {test_loss}')
        print(f'Epoch {epoch}, Val Acc: {val_acc}, Val Loss: {val_loss}')
        
        wandb.log({
            "Test Acc": test_acc, 
            "Test Loss": test_loss, 
            "Training Loss": loss.item(), 
            "Val Loss": val_loss, 
            "Val Acc": val_acc,
            "CE Loss": ce_loss.item(),
            "KL Loss": kl_loss.item(),
            "Combined Loss": loss.item()
        })
        
    return student_model, test_acc, test_loss, val_acc, val_loss

def train_functional(lr=1e-4, wd=0.1, epochs=35, label_smoothing=0.0, warm_up=500, 
                    dataset="svhn", kd_weight=0.5, temperature=2.0):
    """
    Main function for knowledge distillation with LoRA fine-tuning
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define hyperparameter configuration
    hyper_param_config = {
        "dataset": dataset,
        "lr": lr,
        "wd": wd,
        "epochs": epochs,
        "label_smoothing": label_smoothing,
        "warm_up": warm_up,
        "kd_weight": kd_weight,  # Weight for KD loss
        "temperature": temperature  # Temperature for softening distributions
    }
    
    print(hyper_param_config)
    wandb.init(project='KD-LoRA-Distillation', config=hyper_param_config)
    wandb.log({'dataset_name': dataset})
    
    # Configure LoRA for student model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none"
    )
    
    wandb.log(lora_config.__dict__)
    
    # Load student model (ViT-B/32 with LoRA)
    student_path = "openai/clip-vit-base-patch32"
    student_model = HFLoRACLIPVisionModel(
        model_name=student_path,
        cache_dir='/nethome/becsedi3/flash/logs/kd_distill/cache',
        lora_config=lora_config.__dict__,
        device=device
    )
    
    # Load teacher model (ViT-L/14 with LoRA)
    teacher_path = "openai/clip-vit-large-patch14"
    teacher_model = HFLoRACLIPVisionModel(
        model_name=teacher_path,
        cache_dir='/nethome/becsedi3/flash/logs/kd_distill/cache',
        device=device
    )
    
    # Get the specific fine-tuned LoRA weights for the chosen dataset
    # Find the path in the config file based on the dataset name
    config_name = 'multidataset_hf_clip'
    raw_config = get_config_from_name(config_name, device=device)
    
    # Find the appropriate teacher model checkpoint for the selected dataset
    teacher_checkpoint = None
    for base_path in raw_config['model']['bases']:
        if dataset in base_path and 'ViT-L14' in base_path:
            teacher_checkpoint = base_path
            break
    
    if teacher_checkpoint is None:
        raise ValueError(f"No suitable teacher checkpoint found for dataset {dataset}")
    
    print(f"Loading teacher checkpoint: {teacher_checkpoint}")
    teacher_state_dict = torch.load(teacher_checkpoint, map_location=device)
    teacher_model.load_state_dict(teacher_state_dict, strict=False)
    
    # Prepare dataset and preprocess
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])
    
    for dataset_config in raw_config['dataset']:
        dataset_config['train_preprocess'] = student_model.train_preprocess
        dataset_config['eval_preprocess'] = student_model.val_preprocess
    
    data_loaders = prepare_data(raw_config['dataset'], device=device)
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    
    # Create directory for saving models
    model_save_dir = '/nethome/becsedi3/flash/model-checkpoints/kd_lora_multidataset'
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Train on selected dataset
    val_loss = 0
    for dataset_name, loader_dict, class_vectors in tqdm(zip(dataset_names, data_loaders, all_clip_encodings)):
        if dataset_name != dataset:
            continue
        
        print(f'Knowledge Distillation LoRA on {dataset_name}')
        
        # Create a fresh student model for training
        student_lora_model = deepcopy(student_model)
        
        # Freeze the teacher model
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Train with knowledge distillation
        finetuned_model, test_acc, test_loss, val_acc, val_loss = train_cliphead_kd_lora(
            student_model=student_lora_model,
            teacher_model=teacher_model,
            train_loader=loader_dict['train']['full'],
            val_loader=loader_dict['test']['val'],
            test_loader=loader_dict['test']['test'],
            class_vectors=class_vectors,
            hyper_param_config=hyper_param_config
        )
        
        # Save the distilled model
        save_path = os.path.join(
            model_save_dir,
            f'KD_{dataset_name}_lr_{lr}_epochs_{epochs}_wd_{wd}_alpha_{kd_weight}_temp_{temperature}_rank{lora_config.r}.pt'
        )
        
        save_model(finetuned_model, save_path)
        print(f'Model Saved at {save_path}!!')
    
    wandb.finish()
    return val_loss

if __name__ == "__main__":
    # Task-specific configurations
    task_epochs = {
        'stanford_cars': 15,
        'dtd': 15,
        'eurosat': 8,
        'gtsrb': 5,
        'mnist': 2,
        'resisc45': 7,
        'sun397': 8,
        'svhn': 5,
    }
    
    # Select the dataset to train on
    dataset = "svhn"
    
    # Hyperparameter search ranges
    lr_range = [3e-2, 3e-3, 3e-4, 3e-5]
    wd_range = [1e-4, 1e-5]
    kd_weight_range = [0.25, 0.5, 0.75]  # Weight for knowledge distillation
    temperature_range = [2.0]  # Temperature for softening
    
    for wd in wd_range:
        for lr in lr_range:
            for kd_weight in kd_weight_range:
                for temperature in temperature_range:
                    loss = train_functional(
                        lr=lr,
                        wd=wd,
                        epochs=task_epochs[dataset],
                        label_smoothing=0.0,
                        dataset=dataset,
                        kd_weight=kd_weight,
                        temperature=temperature
                    )
                    print(f"Final validation loss: {loss}")
