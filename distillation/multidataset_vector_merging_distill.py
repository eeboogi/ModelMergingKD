import os
import pdb
import torch
from copy import deepcopy

import numpy as np
from itertools import product
from utils import *
from task_merger import TaskMerger
from itertools import combinations
from task_merger import get_merge_handler
from itertools import combinations

def run_BIG_function():
    EVAL_SPLIT = 'val'
    AUX_INFO = ''
    EVAL_TEST = True
    BIGSEED = 420 #421, 422, 1024
    
    print("Seed : ", BIGSEED)
    set_seed(BIGSEED)
     
    task_list = ['stanford_cars', 'dtd', 'eurosat', 'gtsrb', 'mnist', 'resisc45', 'sun397', 'svhn']
    tasks_to_merge = 2
    randSubselect = False
    randomSampleSize = 28 #param for randSubselect
    print(f"Incremental merge over {tasks_to_merge} tasks")
    
    #pdb.set_trace()
    
    merge_indices_all = list(combinations(range(0, 8), tasks_to_merge))
    merge_indices = []
    for index in merge_indices_all:
        if 7 in index:
            merge_indices.append(index)
    
    #pdb.set_trace()
    print("Distilled merge results")
    print(merge_indices)
    
    if randSubselect and len(merge_indices) > randomSampleSize:
        print(f"Randomly selecting {randomSampleSize} runs from all possible task combinations.")
        indices = random.sample(range(len(merge_indices)), randomSampleSize)
        indices.sort()
        print("Random indices: ", indices)
        chosen_indices = [merge_indices[i] for i in indices]
        print(chosen_indices)
        merge_indices = chosen_indices
    
    num_merges = len(merge_indices)
    
    # IMPORTANT NOTE: make sure that in the config file, dataset configs, paths and anything dataset related are ordered in the same order as above in task_list
    config_name = 'multidataset_hf_clip'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(config_name, device=device)
    print(f"Merge method: {raw_config['task_merge_config']['merge_method']}.")
    print(f"Representation: {raw_config['task_merge_config']['representation']}.")
    
    master_results = []
    
    idx_to_continue = len(master_results)
    #pdb.set_trace()
    
    for curr_idx, curr_merge_indices in enumerate(merge_indices):
        
        #continue from where we left off
        if curr_idx < idx_to_continue:
            continue
        
        master_results.append({curr_idx: {}})
        curr_merge_indices = list(curr_merge_indices)
        curr_task_list = [task_list[i] for i in curr_merge_indices]
        print(f"Idx: {curr_idx}/{num_merges-1}. Starting the merge for {curr_task_list}.")
        
        # Get clip encodings
        all_clip_encodings = [get_clip_encodings(raw_config['dataset'][i]['clip_encodings']) for i in curr_merge_indices]
        config = prepare_experiment_config(raw_config)
        dataset_names = np.array([raw_config['dataset'][i]['name'] for i in curr_merge_indices])
        # print('Merging Function: {}'.format(config['merging_fn']))
        dataloaders = np.array([config['data'][i] for i in curr_merge_indices]) #this should be just the subset!
        
        transform_listified = [str(i) for i in list(raw_config['task_merge_config'].values())]
        transform_listified += [str(v) for k, v in raw_config['model']['ft_config'].items() if k in {'r', 'type', 'lora_alpha'}]
        
        csv_file = os.path.join(
            './csvs',
            ":".join(dataset_names),
            raw_config['model']['name'],
            raw_config['eval_type'],
            # "-".join([raw_config['task_merge_config']['merge_method'], raw_config['task_merge_config']['representation']]),
            ":".join(transform_listified),
            f'{EVAL_SPLIT}_{AUX_INFO}.csv'
        )
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        print(f'Saving results to {csv_file}')

        # Parameters are tuned in the order specified in search_config
        # TODO: make selection, comment out rest
        
        # DARE Search Config
        #default_params = {'scaling_coeffs': 1.0, 'topK' : 100, 'pre_svd_topK' : 100, 'interference_threshold': 0.0, 'dare_pruning_coeffs': 0.9} #Default config for DARE
        #order_of_processing_params = ['scaling_coeffs', 'dare_pruning_coeffs'] #DARE-TIES, KnOTS-DARE-TIES, DARE-TV
        
        # TIES, KnOTS-TIES Search Config
        default_params = {'scaling_coeffs': 1.0, 'topK' : 30, 'pre_svd_topK' : 100, 'interference_threshold': 0.0} #Default config for KnOTS/KnOTS-TIES
        order_of_processing_params = ['scaling_coeffs', 'topK'] #TIES, KnOTS-TIES
        
        # TA, KnOTS-TA Search Config
        #default_params = {'scaling_coeffs': 1.0, 'topK' : 30, 'pre_svd_topK' : 100, 'interference_threshold': 0.0}
        #order_of_processing_params = ['scaling_coeffs'] #TV, KnOTS-TV
        
        search_config = {
            # TODO: make selection, comment out rest
            
            # DARE search params
            #'scaling_coeffs': np.arange(0.1, 1.1, step=0.1), 
            #'dare_pruning_coeffs': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, .99], #TODO: uncomment
            
            # TIES, KnOTS-TIES search config
            'scaling_coeffs': np.arange(0.1, 1.1, step=0.1), 
            'topK': (np.arange(1, 11, step=1) * 10)[::-1], #TODO: uncomment
            
            # TA, KnOTS-TA search config
            #'scaling_coeffs': np.arange(0.1, 1.1, step=0.1), 
            
            # 'interference_threshold': np.arange(0.0, 1.1, step=0.1),
            # 'pre_svd_topK': (np.arange(1, 11, step=1) * 10),
            #'scaling_coeffs': np.arange(0.1, 2.1, step=0.1),
            # 'scaling_coeffs': np.arange(0.15, 0.2, step=0.05),
            # 'scaling_coeffs': [1.],
            # 'scaling_coeffs': [0.2],
            # 'scaling_coeffs': [0.2],
            # 'scaling_coeffs':[[1., 0., 0., 0., 0., 0., 0., 0.]],# np.arange(0., 2.0, step=0.1).reshape(-1, 1),
            # 'sign_resolve_mode': ['sum_of_values'],
            # 'sign_resolve_mode': ['sum_signs', 'sign_of_sum'],
            # 'ptm_usage': ['None'],
            # 's_on_V': [False],
            # 'mask_type': ['layer', 'global'],
            # 'sim_type': ['values', 'signs']
        }
        
        #Fine-tuned accuracy to be used to calculate the normalized accuracy
        
        fine_tuned_acc_rank16 = {
            'stanford_cars' : 74.0,
            'dtd' : 58.3,
            'eurosat' : 99.0,
            'gtsrb' : 92.7,
            'mnist' : 99.3,
            'resisc45' : 88.4,
            'sun397' : 64.5,
            'svhn' : 96.2
        }
        
        # fine_tuned_acc_rank128 = {
        #     'stanford_cars' : 77.3,
        #     'dtd' : 68.8,
        #     'eurosat' : 98.5,
        #     'gtsrb' : 96.9,
        #     'mnist' : 99.6,
        #     'resisc45' : 92.2,
        #     'sun397' : 67.4,
        #     'svhn' : 97.1
        # }
        
        # fine_tuned_acc_fft = {
        #     'stanford_cars' : 77.7,
        #     'dtd' : 79.4,
        #     'eurosat' : 99.7,
        #     'gtsrb' : 98.7,
        #     'mnist' : 99.7,
        #     'resisc45' : 96.1,
        #     'sun397' : 75.3,
        #     'svhn' : 97.5
        # }
        
        """
        if config['model']['ft_config']['r'] == 128:
            print("Using rank128 acc to normalize")
            fine_tuned_acc = fine_tuned_acc_rank128
        elif config['model']['ft_config']['r'] == 16:
            print("Using rank16 acc to normalize")
            fine_tuned_acc = fine_tuned_acc_rank16
        elif config['model']['ft_config']['type'] == 'fft':
            print("Using fft fine-tuned acc")
            fine_tuned_acc = fine_tuned_acc_fft
        """
        
        fine_tuned_acc = fine_tuned_acc_rank16 # changed for now
        
        #print(f'Finetuned Accs: {fine_tuned_acc}')
        #print(search_config)
        param_names, values = zip(*search_config.items())
        # pdb.set_trace()
        def merge_and_eval(Merge, EVAL_SPLIT = 'val', instance_params = None):
            set_seed(BIGSEED)
            print("EVAL_SPLIT : ", EVAL_SPLIT)
            print(f'Search Run with: {instance_params}')
            all_results = deepcopy(instance_params)
            print('Creating Merge')
            # iniitalize merging function
            # pdb.set_trace()
            # MergeClass = get_merge_handler(config['task_merge_config']['representation'])
            # Merge = MergeClass(
            #     deepcopy(models), 
            #     pretrained_model=deepcopy(config['models']['new']), 
            #     param_handler=config['param_handler'],
            #     device=device,
            #     merge_config=config['task_merge_config'],
            # )
            # set task scaling coefficients
            Merge.set_scaling_coeffs(instance_params['scaling_coeffs'])
            config['task_merge_config'].update(instance_params)

            merged_model = Merge.merge(config['task_merge_config'])
            # merged_model = Merge.merge_with_same_scale(config['task_merge_config'])
            # merged_model = Merge.apply_ties_on_U_and_V(config['task_merge_config'])
            
            print('Evaluate Merged Model on Each Dataset')
            avg_accuracy = 0.
            avg_norm_accuracy = 0.
            for i, loader_dict in enumerate(dataloaders):
                loader = loader_dict['test'][EVAL_SPLIT]
                acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=all_clip_encodings[i].to(device))
                print(f"{dataset_names[i]} Normalized accuracy is {np.round((acc * 100)/ fine_tuned_acc[dataset_names[i]] *100, 3)}")
                print(f"{dataset_names[i]} accuracy is {np.round(acc * 100, 3)}")
                all_results[dataset_names[i]] = acc * 100
                all_results[dataset_names[i]+'_norm_acc'] = (acc * 100) / fine_tuned_acc[dataset_names[i]] *100
                avg_accuracy += acc * 100
                avg_norm_accuracy += (acc * 100)/ fine_tuned_acc[dataset_names[i]] *100
            avg_accuracy /= len(dataloaders)
            avg_norm_accuracy /= len(dataloaders)
            print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
            print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')
            all_results['Average_acc'] = avg_accuracy
            all_results['Average_norm_acc'] = avg_norm_accuracy
            all_results.update(config['task_merge_config'])
            write_to_csv(all_results, csv_file)
            return all_results
            


        with torch.no_grad():
        
            #print(search_config)
            param_names, values = zip(*search_config.items())

            models = np.array([config['models']['bases'][i].cpu().eval() for i in curr_merge_indices])

            MergeClass = get_merge_handler(config['task_merge_config']['representation'])
            Merge = MergeClass(
                    deepcopy(models), 
                    pretrained_model=deepcopy(config['models']['new']), 
                    param_handler=config['param_handler'],
                    device=device,
                    merge_config=config['task_merge_config'],
                )
            Merge.transform(config['task_merge_config'])
            #print(config['task_merge_config'])
            for param in order_of_processing_params:
                best_val_results = {'Average_norm_acc' : 0.0}
                for value in search_config[param]:
                    instance_params = deepcopy(default_params)
                    instance_params[param] =  value
                    # pdb.set_trace()
                    all_results = merge_and_eval(Merge, EVAL_SPLIT = 'val', instance_params = instance_params)
                    if (all_results['Average_norm_acc'] > best_val_results['Average_norm_acc']):
                        best_val_results = deepcopy(all_results)
                    else:
                        break
                default_params[param] = best_val_results[param]
                master_results[curr_idx] = {"val": best_val_results}
                
            # for full grid search:
            # for bundle in product(*values):
            #     #pdb.set_trace()
            #     print(bundle)
            # best_val_results = {'Average_norm_acc' : 0.0}
            # for bundle in product(*values):
            #     # pdb.set_trace()
            #     instance_params = dict(zip(param_names, bundle))
            #     all_results = merge_and_eval(Merge, EVAL_SPLIT = 'val', instance_params =instance_params)
            #     if (all_results['Average_norm_acc'] > best_val_results['Average_norm_acc']):
            #         best_val_results = deepcopy(all_results)

            if (EVAL_TEST == True):
                # Evaluate on the test set with the best topK and scaling co-efficient
                print("Best params (best_val_results):", best_val_results)
                for key in search_config.keys():
                    instance_params.update({key : best_val_results[key]})
                test_result = merge_and_eval(Merge, EVAL_SPLIT = 'test', instance_params =instance_params)
                print("Test results: ", test_result)
                master_results[curr_idx] = {"val": best_val_results, "test": test_result}
        
        print(master_results)
    
    average_val_acc = 0.0
    average_test_acc = 0.0
    average_val_norm_acc = 0.0
    average_test_norm_acc = 0.0
    
    for run in master_results:
        average_val_acc += run['val']['Average_acc']
        average_test_acc += run['test']['Average_acc']
        average_val_norm_acc += run['val']['Average_norm_acc']
        average_test_norm_acc += run['test']['Average_norm_acc']
    
    average_val_acc /= len(master_results)
    average_test_acc /= len(master_results)
    average_val_norm_acc /= len(master_results)
    average_test_norm_acc /= len(master_results)
    
    return master_results, average_val_acc, average_test_acc, average_val_norm_acc, average_test_norm_acc
            
if __name__ == "__main__":
    
    master_results, avg_val_acc, avg_test_acc, avg_val_norm_acc, avg_test_norm_acc = run_BIG_function()
    print(master_results)
    print("Final results: ")
    
    print("Average val accuracy: ", avg_val_acc)
    print("Average test accuracy: ", avg_test_acc)
    print("Average val norm accuracy: ", avg_val_norm_acc)
    print("Average test norm accuracy: ", avg_test_norm_acc)