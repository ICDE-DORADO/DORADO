import os
import numpy as np
import torch
import argparse
import pdb
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import wandb
import logging
from datetime import datetime
import sys
import utils
from torch_geometric.data import Data
import pickle
import gzip
# dataloader
# from torch_geometric.data import DataLoader

## load data

def re_spilt(dataset, split_ratio=[0.5, 0.2, 0.3]):
    all_data = np.concatenate([dataset.train, dataset.valid, dataset.test], axis=0)
    # np.random.shuffle(all_data)m
    times = np.unique(all_data[:, 3])
    # sort
    times = np.sort(times)
    train_time_deadline = times[int(len(times)*split_ratio[0])]
    valid_time_deadline = times[int(len(times)*(split_ratio[0]+split_ratio[1]))]
    train_data = all_data[all_data[:, 3] <= train_time_deadline]
    valid_data = all_data[(all_data[:, 3] > train_time_deadline) & (all_data[:, 3] <= valid_time_deadline)]
    test_data = all_data[all_data[:, 3] > valid_time_deadline]
    dataset.train = train_data
    dataset.valid = valid_data
    dataset.test = test_data
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Data preprocess for TransFIR")
    parser.add_argument("--dataset", type=str, default="ICEWS14", help="Dataset name (default: ICEWS14)")
    parser.add_argument("--max_length", type=int, default=150, help="Max length of one-hop chain (default: 150)")
    parser.add_argument("--T", type=int, default=14, help="Time period for history (default: 7)")
    args = parser.parse_args()
    return args

def main(args):
    dataset_name = args.dataset
    T = args.T
    split_ratio = [0.5, 0.2, 0.3]
    max_length = args.max_length
    data_base_dir = utils.resolve_data_base_dir(dataset_name)
    data = utils.load_data(dataset_name)
    data = re_spilt(data, split_ratio)
    train_triple = data.train
    valid_triple = data.valid
    test_triple = data.test
    all_triple = np.concatenate([train_triple, valid_triple, test_triple], axis=0)
    times = np.unique(all_triple[:, 3])
    all_entities = np.concatenate([all_triple[:, 0], all_triple[:, 2]])
    all_entities = np.unique(all_entities)
    all_relations = np.unique(all_triple[:, 1])
    all_triple_inverse = all_triple[:, [2, 1, 0, 3]]
    all_triple_inverse[:, 1] += len(all_relations)
    all_triple = np.concatenate([all_triple, all_triple_inverse], axis=0)

    known_entities = set(train_triple[:, 0]) | set(train_triple[:, 2])
    unknown_entities = set(all_entities) - known_entities
    print(f"Known entities: {len(known_entities)}, Unknown entities: {len(unknown_entities)}, Time steps: {len(times)}")
    entity_history = {}
    for entity in all_entities:
        triple_with_entity = all_triple[all_triple[:, 0] == entity]
        triple_with_entity_sort_by_time = triple_with_entity[np.argsort(triple_with_entity[:, 3])]
        entity_history[entity] = triple_with_entity_sort_by_time


    class TKGDataset_for_data_preparation(Dataset):
        def __init__(self, data_dict, all_triples, data_type='sro'):
            self.data_dict = data_dict
            self.triples = all_triples
            self.times = np.unique(all_triples[:, 3])
            self.num_relations = np.unique(all_triples[:, 1]).shape[0]
            self.num_relations = int(self.num_relations / 2)
            self.type = data_type
            self.max_time_period = T
            self.one_hop_max_length = 150
        def __len__(self):
            return len(self.times)
        
        def __getitem__(self, index):
            time = self.times[index]
            triples_at_time = self.triples[self.triples[:, 3] == time]
            path_dict = {}
            for idx in range(len(triples_at_time)):
                triple = triples_at_time[idx]
                s, r, o, t = triple
                history = self.data_dict[s]
                history = history[history[:, 3] < t]
                if index >= self.max_time_period:
                    history = history[history[:, 3] >= self.times[index-self.max_time_period]]

                if len(history) > self.one_hop_max_length:
                    history = history[-self.one_hop_max_length:]

                # Keep the legacy key for backward compatibility while using a more neutral name.
                result = {'context_window': history, 'one_hop_chain': history}
                path_dict[idx] = result

            return path_dict
        

    dataset = TKGDataset_for_data_preparation(entity_history, all_triple, max_length)

    # Use compression and highest protocol to reduce memory usage
    output_dir = os.path.join(data_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_name}_T_{T}.pkl')
    
    print(f"\nProcessing {len(times)} time steps...")
    all_result = {}
    for time in tqdm(range(len(times)), desc="Processing time steps"):
        all_result[time] = dataset[time]
    
    # Use gzip compression and HIGHEST_PROTOCOL for better memory efficiency
    print(f"\n{'='*60}")
    print(f"Saving processed data to: {output_file}")
    print(f"Data size: {len(all_result)} time steps")
    print("This may take 5-15 minutes due to:")
    print("  - Large data size (~2-4 GB in memory)")
    print("  - Pickle serialization")
    print("  - Gzip compression")
    print("Please wait, the program is working...")
    print(f"{'='*60}\n")
    
    import time
    start_time = time.time()
    
    # Use compresslevel=1 for faster compression (trade-off: larger file size)
    with gzip.open(output_file, 'wb', compresslevel=1) as f:
        pickle.dump(all_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully saved to: {output_file}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Save time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
