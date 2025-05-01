import os
import torch
from copy import deepcopy
import pickle
import numpy as np
from utils import *
from task_merger import get_merge_handler
from peft import LoraConfig
from models.huggingface_clip import HFLoRACLIPVisionModel

def run_distilled_svhn_merging_experiment():
    EVAL_SPLIT = 'val'
    AUX_INFO = 'distilled_svhn_experiment'
    EVAL_TEST = True
    BIGSEED = 420

    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
    
    # Load configuration
    config_name = 'multidataset_hf_clip_kd_merge'  # New config file with distillation specs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    
    # Get clip encodings
    all_clip_encodings = [get_clip_encodings(i['clip_encodings']) for i in raw_config['dataset']]
    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([i['name'] for i in raw_config['dataset']])

    dataloaders = np.array([i for i in config['data']])
    
    transform_listified = [str(i) for i in list(raw_config['task_merge_config'].values())]
    transform_listified += [str(v) for k, v in raw_config['model']['ft_config'].items() if k in {'r', 'type', 'lora_alpha'}]
    
    csv_file = os.path.join(
        './csvs',
        ":".join(dataset_names),
        raw_config['model']['name'],
        raw_config['eval_type'],
        ":".join(transform_listified),
        f'{EVAL_SPLIT}_{AUX_INFO}.csv'
    )
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    print(f'Saving results to {csv_file}')

    # Parameters for merging
    default_params = {
        'scaling_coeffs': np.ones(2) * .2,  # For pairwise merging
        'topK': 100, 
        'pre_svd_topK': 100, 
        'interference_threshold': 0.0
    }
    
    order_of_processing_params = [
        'scaling_coeffs', 
        'topK',
    ]
    
    search_config = {
        'scaling_coeffs': np.arange(0.1, 1.1, step=0.1),
        'topK': (np.arange(1, 11, step=1) * 10)[::-1],
    }
    
    # Task names and indices
    model_order = [
        'mnist',
        'eurosat',
        'svhn',
        'gtsrb',
        'resisc45',
        'stanford_cars',
        'sun397',
        'dtd'
    ]
    
    svhn_idx = model_order.index('svhn')
    other_task_indices = [i for i, name in enumerate(model_order) if name != 'svhn']
    
    fine_tuned_acc_rank16 = {
        'stanford_cars': 74.0,
        'dtd': 58.3,
        'eurosat': 99.0,
        'gtsrb': 92.7,
        'mnist': 99.3,
        'resisc45': 88.4,
        'sun397': 64.5,
        'svhn': 96.2
    }
    
    print(f'Finetuned Accs: {fine_tuned_acc_rank16}')
    print(search_config)
    
    # Function to merge and evaluate models
    def merge_and_eval(models_to_merge, task_indices, EVAL_SPLIT='val', instance_params=None):
        set_seed(BIGSEED)
        print("EVAL_SPLIT : ", EVAL_SPLIT)
        print(f'Search Run with: {instance_params}')
        all_results = deepcopy(instance_params)
        
        # Initialize merger
        MergeClass = get_merge_handler(config['task_merge_config']['representation'])
        Merge = MergeClass(
            models_to_merge, 
            pretrained_model=deepcopy(config['models']['new']), 
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        
        Merge.transform(config['task_merge_config'])
        
        # Set scaling coefficients
        Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
        config['task_merge_config'].update(instance_params)

        # Merge the models
        merged_model = Merge.merge(config['task_merge_config'])
        
        print('Evaluate Merged Model on Each Dataset')
        avg_accuracy = 0.
        avg_norm_accuracy = 0.
        total_used = 0
        
        # Evaluate on the tasks that were merged
        for i in task_indices:
            total_used += 1
            loader = dataloaders[i]['test'][EVAL_SPLIT]
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
            print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ fine_tuned_acc_rank16[dataset_names[i]] *100, 3)}")
            print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
            all_results[dataset_names[i]] = acc * 100
            all_results[dataset_names[i]+'_norm_acc'] = (acc * 100) / fine_tuned_acc_rank16[dataset_names[i]] * 100
            avg_accuracy += acc * 100
            avg_norm_accuracy += (acc * 100) / fine_tuned_acc_rank16[dataset_names[i]] * 100
        
        avg_accuracy /= total_used
        avg_norm_accuracy /= total_used
        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
        all_results['Average_acc'] = avg_accuracy
        all_results['Average_norm_acc'] = avg_norm_accuracy
        all_results.update(config['task_merge_config'])
        write_to_csv(all_results, csv_file)
        return all_results
    
    # Function to tune hyperparameters and evaluate
    def tune_and_evaluate(svhn_model, other_task_idx, EVAL_TEST=True):
        other_task = model_order[other_task_idx]
        print(f"\nMerging svhn with {other_task}")
        
        # Create pair of models
        pair_models = np.array([deepcopy(svhn_model), deepcopy(all_models[other_task_idx])])
        
        # Tune hyperparameters
        instance_params = deepcopy(default_params)
        for param in order_of_processing_params:
            best_val_results = {'Average_norm_acc': 0.0}
            for value in search_config[param]:
                instance_params[param] = value
                all_results = merge_and_eval(pair_models, [svhn_idx, other_task_idx], EVAL_SPLIT='val', instance_params=instance_params)
                if all_results['Average_norm_acc'] >= best_val_results['Average_norm_acc']:
                    best_val_results = deepcopy(all_results)
            instance_params[param] = best_val_results[param]
        
        # Evaluate with best parameters on test set
        if EVAL_TEST:
            print("Best params:", instance_params)
            test_result = merge_and_eval(pair_models, [svhn_idx, other_task_idx], EVAL_SPLIT='test', instance_params=instance_params)
            return test_result
        return best_val_results
    
    # Load all models
    with torch.no_grad():
        # Define LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.1,
            bias="none"
        )
        
        # Initialize and load all the base models
        all_models = []
        for model_path in config['model']['bases']:
            # Create a new model instance
            model = HFLoRACLIPVisionModel(
                model_name=config['model']['base_type'],
                cache_dir=config['model']['cachedir'],
                lora_config=lora_config.__dict__,
                device='cpu'  # Load on CPU first
            )
            
            # Load weights
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            
            all_models.append(model)
        
        all_models = np.array(all_models)
        
        # Load the distilled svhn model
        distilled_model_path = raw_config['distill_config']['distilled_model_path']
        print(f"Loading distilled model from {distilled_model_path}")
        
        # Create new model instance for distilled model
        distilled_svhn_model = HFLoRACLIPVisionModel(
            model_name=config['model']['base_type'],
            cache_dir=config['model']['cachedir'],
            lora_config=lora_config.__dict__,
            device='cpu'
        )
        
        # Load weights
        distilled_state_dict = torch.load(distilled_model_path, map_location='cpu')
        distilled_svhn_model.load_state_dict(distilled_state_dict, strict=False)
        distilled_svhn_model.eval()
        
        # Store results
        distilled_results = {}
        regular_results = {}
        
        # For each task (except svhn), merge with both distilled and regular svhn models
        for other_idx in other_task_indices:
            # First run with distilled svhn model
            print("\n=== Using DISTILLED svhn model ===")
            distilled_results[model_order[other_idx]] = tune_and_evaluate(distilled_svhn_model, other_idx)
            
            # Then run with regular svhn model (baseline)
            print("\n=== Using REGULAR svhn model (baseline) ===")
            regular_results[model_order[other_idx]] = tune_and_evaluate(all_models[svhn_idx], other_idx)
        
        # Save all results
        results = {
            'distilled_svhn': distilled_results,
            'regular_svhn': regular_results
        }
        
        results_file = os.path.join(os.getcwd(), "distilled_svhn_merging_results.pkl")
        print(f"\nSaving results to {results_file}")
        pickle.dump(results, open(results_file, 'wb'))
        
        # Print summary of results
        print("\n=== RESULTS SUMMARY ===")
        print("Average normalized accuracy for each pair:")
        for task in [t for t in model_order if t != 'svhn']:
            dist_acc = distilled_results[task]['Average_norm_acc']
            reg_acc = regular_results[task]['Average_norm_acc']
            diff = dist_acc - reg_acc
            print(f"svhn + {task}: Distilled: {dist_acc:.2f}, Regular: {reg_acc:.2f}, Diff: {diff:.2f}")

if __name__ == "__main__":
    print("Running distilled svhn merging experiment")
    run_distilled_svhn_merging_experiment()