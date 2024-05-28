# Save this as umls_pretrain.py

import os
import torch
import random
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
sys.path.insert(0, r"C:\Users\dlais\OneDrive - University of Illinois Chicago\Desktop\rescnn_bioel-adapter-pretraining - Copy")
os.chdir(r"C:\Users\dlais\OneDrive - University of Illinois Chicago\Desktop\rescnn_bioel-adapter-pretraining - Copy")
from tqdm import tqdm
from utils import get_n_params, create_dir_if_not_exist, get_n_tunable_params, AugmentedList, RunningAverage
from models import AdapterModel
from data.base import Ontology, DataInstance, PretrainingPositivePairs
from scripts import benchmark_model
from argparse import ArgumentParser
from constants import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def create_dir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
def pretrain(configs):
    # Load model
    model = AdapterModel(configs)
    print(f'Prepared the model (Params: {get_n_params(model)})')
    print(f'Nb Tunable Params: {get_n_tunable_params(model)}')

    # Load UMLS-2020 AA Full Ontology
    ontology_path = configs['ontology_path']  # Correctly access the ontology path
    ontology = Ontology(ontology_path)
    print('Loaded UMLS-2020 AA Full Ontology')

    # Prepare the train set
    train, example_id = [], 0
    positive_pairs = PretrainingPositivePairs(configs['positive_pairs_path'])  # Correctly access the pairs path
    for n1, n2 in positive_pairs:
        mention = {
            'term': n1.name_str,
            'entity_id': n1.entity_id
        }
        inst = DataInstance(example_id, '', mention)
        inst.selected_positive = n2
        train.append(inst)
        example_id += 1
    random.shuffle(train)
    train = AugmentedList(train, shuffle_between_epoch=True)
    print(f'Train size: {len(train)}')

    # Prepare the optimizer and the scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    num_epoch_steps = math.ceil(len(train) / configs['batch_size'])
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start Training
    print('Batch size: {}'.format(configs['batch_size']))
    print('Epochs: {}'.format(configs['epochs']))
    iters, batch_loss, best_score = 0, 0, 0
    batch_size = configs['batch_size']
    accumulated_loss = RunningAverage()
    gradient_accumulation_steps = configs['gradient_accumulation_steps']
    for epoch in range(configs['epochs']):
        with tqdm(total=num_epoch_steps, desc=f'Epoch {epoch}') as pbar:
            for _ in range(num_epoch_steps):
                iters += 1
                instances = train.next_items(batch_size)

                # Compute iter_loss
                iter_loss = model(instances, ontology, is_training=True)[0]
                iter_loss = iter_loss / gradient_accumulation_steps
                iter_loss.backward()
                batch_loss += iter_loss.data.item()

                # Update params
                if iters % gradient_accumulation_steps == 0:
                    accumulated_loss.update(batch_loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss = 0

                # Update pbar
                pbar.update(1)
                pbar.set_postfix_str(f'Iters: {iters} Student Loss: {accumulated_loss()}')

                # Evaluation and Model Saving
                if (iters % 5000 == 0) or (iters % num_epoch_steps == 0):
                    print(f'{iters} Benchmarking model')
                    model_score = benchmark_model(model, batch_size, [BC5CDR_C, BC5CDR_D, NCBI_D, COMETA], 'test')
                    print('Overall model score: {}'.format(model_score))
                    if model_score > best_score:
                        best_score = model_score
                        # Ensure the save directory exists before saving
                        create_dir_if_not_exist(configs['save_dir'])
                        model.save_pretrained(configs['save_dir'])  # Save the full model

                    print('', flush=True)

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--no_cuda', type=str2bool, default=False)
    parser.add_argument('--report_frequency', type=int, default=100)
    parser.add_argument('--epoch_evaluation_frequency', type=int, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--feature_proj', type=str2bool, default=False)
    parser.add_argument('--feature_size', type=int, default=768)
    parser.add_argument('--gradient_checkpointing', type=str2bool, default=False)
    parser.add_argument('--max_length', type=int, default=25)
    parser.add_argument('--use_synthetic_train', type=str2bool, default=False)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--transformer_learning_rate', type=float, default=2e-05)
    parser.add_argument('--task_learning_rate', type=float, default=0.0001)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--loss_scale_pos', type=int, default=2)
    parser.add_argument('--loss_scale_neg', type=int, default=50)
    parser.add_argument('--loss_thresh', type=float, default=0.5)
    parser.add_argument('--loss_lambda', type=float, default=0.2)
    parser.add_argument('--online_kd', type=str2bool, default=False)
    parser.add_argument('--gradual_unfreezing', type=str2bool, default=False)
    parser.add_argument('--lightweight', type=str2bool, default=False)
    parser.add_argument('--hard_negatives_training', type=str2bool, default=False)
    parser.add_argument('--transformer', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset', type=str, default='UMLS-2020AA-Full')
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--ontology_path', type=str, default=r"C:\Users\dlais\OneDrive - University of Illinois Chicago\Desktop\BioNLP\umls-2023AB-metathesaurus-full\2023AB\umls_dict.json")  # Add this argument
    parser.add_argument('--positive_pairs_path', type=str, default=r"C:\Users\dlais\OneDrive - University of Illinois Chicago\Desktop\BioNLP\Datasets\pairs.txt")  # Add this argument

    args = parser.parse_args()

    # Convert args to a dictionary
    configs = vars(args)

    # Train
    pretrain(configs)