# vit_arch = 'ViT-L-14-CLIP'
vit_arch = 'ViT-B-32-CLIP'

config = {
    'dataset': [
        # {
        #     'name': 'domainNet_clipart',
        #     'shuffle_train': True,
        #     'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/domainNet_clipart_head.pt',
        #     'val_fraction': 0.2
        # },
        # # {
        # #     'name': 'domainNet_infograph',
        # #     'shuffle_train': True,
        # #     'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/domainNet_clipart_head.pt',
        # #     'val_fraction': 0.2
        # # },
        # {
        #     'name': 'domainNet_painting',
        #     'shuffle_train': True,
        #     'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/domainNet_clipart_head.pt',
        #     'val_fraction': 0.2
        # },
        # {
        #     'name': 'domainNet_quickdraw',
        #     'shuffle_train': True,
        #     'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/domainNet_clipart_head.pt',
        #     'val_fraction': 0.2
        # },
        # {
        #     'name': 'domainNet_real',
        #     'shuffle_train': True,
        #     'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/domainNet_clipart_head.pt',
        #     'val_fraction': 0.2
        # },
        # {
        #     'name': 'domainNet_sketch',
        #     'shuffle_train': True,
        #     'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/domainNet_clipart_head.pt',
        #     'val_fraction': 0.2
        # },
        
        {
            'name': 'stanford_cars',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/stanford_cars_head.pt',
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'dtd',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/dtd_head.pt',
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'eurosat',
            # 'train_fraction': 0.5,
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/eurosat_head.pt',
            # 'res': 256,
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'gtsrb',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/gtsrb_head.pt',
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'mnist',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/mnist_head.pt',
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,            
        },
        {
            'name': 'resisc45',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/resisc45_head.pt',
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'sun397',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/sun397_head.pt',
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 16,
        },
        {
            'name': 'svhn',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/' + vit_arch + '/svhn_head.pt',
            'val_fraction': 0.2,
            'batch_size': 32,
            'num_workers': 8,
        },
    ],
    'model': {
        'name': 'hf_clip',
        # 'base_type': "openai/clip-vit-large-patch14",
        'base_type': "openai/clip-vit-base-patch32",
        'cachedir': '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/cache',
        # 'cachedir': '/srv/hoffman-lab/flash9/gstoica3/ModelMerging/ckpts/cache',
        'bases': [
            #DomainNet Models:
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts_old/ckpts/selcted_r6_models/domainNet_clipart_lr_0.001_epochs_100_wd_0.0001_label_smoothing_0.1.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts_old/ckpts/selcted_r6_models/domainNet_painting_lr_0.001_epochs_100_wd_0.0001_label_smoothing_0.1.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts_old/ckpts/selcted_r6_models/domainNet_quickdraw_lr_0.001_epochs_30_wd_0.0001_label_smoothing_0.1.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMergi ng/ckpts_old/ckpts/selcted_r6_models/domainNet_real_lr_0.001_epochs_30_wd_0.0001_label_smoothing_0.1.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts_old/ckpts/selcted_r6_models/domainNet_sketch_lr_0.001_epochs_50_wd_0.0001_label_smoothing_0.1.pt',

            #Zero shot model
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/zeroshot.pt',

            #rank-16 8 vision tasks
            '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/stanford_cars_lora-r16-alpha16-dropout0.1.pt',
            '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/dtd_lora-r16-alpha16-dropout0.1.pt',
            '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/eurosat_lora-r16-alpha16-dropout0.1.pt',
            '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/gtsrb_lora-r16-alpha16-dropout0.1.pt',
            '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/mnist_lora-r16-alpha16-dropout0.1.pt',
            '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/resisc45_lora-r16-alpha16-dropout0.1.pt',
            '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/sun397_lora-r16-alpha16-dropout0.1.pt',
            #'/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/svhn_lora-r16-alpha16-dropout0.1.pt',
            '/nethome/becsedi3/flash/model-checkpoints/kd_lora_multidataset/best_models/KD_svhn_from_mnist_lr_0.0005_epochs_5_wd_5e-05_alpha_0.25_temp_1.0_rank16.pt'

            #ViT-L-14 model, rank-16 models (old location)
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/stanford_cars_lr_0.0003_epochs_15_wd_0.0001_label_smoothing_0.0_rank16.pt',  
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/dtd_lr_0.0003_epochs_15_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/eurosat_lr_0.0003_epochs_8_wd_0.0001_label_smoothing_0.0_rank16.pt',  
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/gtsrb_lr_0.0003_epochs_5_wd_0.0001_label_smoothing_0.0_rank16.pt',  
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/mnist_lr_0.0003_epochs_2_wd_0.0001_label_smoothing_0.0_rank16.pt',  
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/resisc45_lr_0.0003_epochs_7_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/sun397_lr_0.0003_epochs_8_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/selected/svhn_lr_0.0003_epochs_5_wd_0.0001_label_smoothing_0.0_rank16.pt',

            #ViT-L-14 model, rank-16 models (new location)
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/stanford_cars_lr_0.0003_epochs_15_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/dtd_lr_0.0003_epochs_15_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/eurosat_lr_0.0003_epochs_8_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/gtsrb_lr_0.0003_epochs_5_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/mnist_lr_0.0003_epochs_2_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/resisc45_lr_0.0003_epochs_7_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/sun397_lr_0.0003_epochs_8_wd_0.0001_label_smoothing_0.0_rank16.pt',
            # '/srv/hoffman-lab/flash9/pramesh39/ModelMerging/ckpts/ViT-L14/fft/arxival/selected/svhn_lr_0.0003_epochs_5_wd_0.0001_label_smoothing_0.0_rank16.pt',
            
            #rank-128 8 vision tasks
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/stanford_cars_lr_0.0013175004430534048_epochs_35_wd_0.017640127338830927_label_smoothing_0.0.pt',
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/dtd_lr_0.0006053899934426028_epochs_76_wd_0.06122764567379299_label_smoothing_0.0.pt',
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/eurosat_lr_0.0018867188898503483_epochs_12_wd_0.00046059170264975934_label_smoothing_0.11787336296491036.pt',
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/gtsrb_lr_0.0010094568837490053_epochs_11_wd_0.000811017906116537_label_smoothing_0.060759049404556634.pt',
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/mnist_lr_0.0009491706294158518_epochs_10_wd_0.0003571129465418084_label_smoothing_0.13724116349715978.pt',
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/resisc45_lr_0.0019464523141699253_epochs_20_wd_0.00033803209321832666_label_smoothing_0.0.pt',
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/sun397_lr_0.000878396333134627_epochs_20_wd_1.8029271527207212e-05_label_smoothing_0.0.pt',
            # '/coc/pskynet4/pramesh39/checkpoints/lora_r128/svhn_lr_0.0012584300556599676_epochs_8_wd_1.8653991180976714e-05_label_smoothing_0.1420089041631571.pt',

            #Orthogonalized models:
            # '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/gtsrb_svhn/gtsrb_lora-r16-alpha16-dropout0.1.pt',
            # '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/gtsrb_svhn/svhn_lora-r16-alpha16-dropout0.1.pt',
            # '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/stanford_cars_svhn/all_but_qk/stanford_cars_lora-r16-alpha16-dropout0.1.pt',
            # '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/stanford_cars_svhn/all_but_qk/svhn_lora-r16-alpha16-dropout0.1.pt'

            #Orthogonal All but QK
            # '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/gtsrb_svhn/all_but_qk/gtsrb_lora-r16-alpha16-dropout0.1.pt',
            # '/coc/pskynet2/gstoica3/checkpoints/multiset/openai/ViT-B-32-CLIP/lora_models/gtsrb_svhn/all_but_qk/svhn_lora-r16-alpha16-dropout0.1.pt',

            #Model soups model directories
            # '/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/dtd'
            #'/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/eurosat'
            #'/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/gtsrb'
            #'/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/mnist'
            #'/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/resisc45'
            #'/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/stanford_cars'
            #'/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/sun397'
            #'/nethome/becsedi3/flash/model-checkpoints/lora_finetuning_multidataset_ckpts/model_soups/svhn'
            
            # From HuggingFace
            #'hoffman-lab/KnOTS-ViT-L-14_lora_R16_mnist',
            #'hoffman-lab/KnOTS-ViT-B-32_lora_R16_mnist'
            
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ["q_proj", "k_proj", "v_proj", "out_proj"],
            'lora_dropout': 0.1,
            'bias': "none",
        },
    },
    'task_merge_config': {
        'representation': 'svd-vector', #vector
        # 'representation': 'svd-method2',
        # 'representation': 'vector',
        'ptm_usage': 'None',
        'sign_resolve_mode': 'sum_of_values',
        's_on_V': True,
        # 'idx': 1,
        'topK': 100,
        'pre_svd_topK' : 100,
        'merge_method': 'ties', #tv, ties
        'mask_method': 'ties',
        'merging_type': 'mean', #mean for ties, sum for tv
        # 'merge_method': 'tv',
        # 'mask_method': 'tv',
        # 'merging_type': 'sum',
        # 'merging_type': 'regmean',
        'scaling_coeffs': .6, #[.6],
        'concat_across_output': True,   # When True: U is of size O x nI; False: U is of size I x nO
        'normalize_Vs_per_model': False,
        'normalize_Vs': None,         # Options: 'cols', 'rows' When cols: V's are normalized across rows/cols
        'normalize_Ws_by': None, # Options: l2, colwise_l2, rowwise_l2, l1, colwise_l1, rowwise_l1, None <-- The actual variable, not a string
        'erase_s': False,
        'merge_other_params' : False,
        'label_normalization' : None,
        #'dare' : None,
        'dare' : None, #DARE: 'Vs', other: None
        'dare_pruning_coeffs' : 0.9,
        'whiten': False,
        # 'matching_method': 'cosine_avg',
        # 'combine_measure': 'cosine_avg'
        'normalize_masked_Vs': False,
        # 'regmean_innerprods_path': '/srv/share2/gstoica3/metric_results/inner_prods/all_eight_datasets/lora_r16.pkl',
        # 'regmean_innerprods_path': '/srv/share2/gstoica3/metric_results/inner_prods_val/all_eight_datasets/lora_r16.pkl'
        'unit_normalize_merge': False,
        # 'prune_by_layer': True,
        # 'prune_by_concated_layers': False,
    },
    'eval_type': 'clip',
    'merging_fn': 'match_tensors_randperm',
    'merging_metrics': ['covariance', 'mean'],
    'finetune_epochs': 0,
    'distill_config': {
        'cache_dir': '/nethome/becsedi3/flash/logs/kd_distill/cache',
        'model_save_dir': '/nethome/becsedi3/flash/model-checkpoints/kd_lora_multidataset',
        'teacher_base_name': "openai/clip-vit-large-patch14",
        'student_base_name': "openai/clip-vit-base-patch32",
        'dataset_name': "svhn",
        'alpha': 0.5, # Weight for CrossEntropy loss (1.0 - alpha for KD loss)
        'temperature': 2.0,
    },
}

