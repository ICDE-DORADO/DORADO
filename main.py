import os
import numpy as np
import torch
import argparse
from model import *
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset
import logging
from datetime import datetime
import sys
import pickle
import random
import gzip

import utils


def r2e(triplets, num_rels):
    from collections import defaultdict
    src, rel, dst = triplets.transpose()
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r + num_rels))
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel + num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx, idx + len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx


def load_pickle_auto(file_path):
    """Load pickle file with optional gzip compression."""
    with open(file_path, 'rb') as f:
        magic = f.read(2)

    if magic == b'\x1f\x8b':
        with gzip.open(file_path, 'rb') as f:
            return pickle.load(f)

    with open(file_path, 'rb') as f:
        return pickle.load(f)


LOSS_COMPONENT_NAMES = ("task_loss", "ortho_loss", "temporal_contrastive_loss", "distill_loss")


def init_loss_components():
    return {name: [] for name in LOSS_COMPONENT_NAMES}


def record_loss_components(loss_components, task_loss, ortho_loss, temporal_contrastive_loss, distill_loss):
    for name, loss_value in zip(
            LOSS_COMPONENT_NAMES,
            (task_loss, ortho_loss, temporal_contrastive_loss, distill_loss)):
        loss_components[name].append(float(loss_value.detach().item()))


def format_loss_components(loss_components):
    return " | ".join(
        f"{name}: {np.mean(values):.4f}" if len(values) > 0 else f"{name}: 0.0000"
        for name, values in loss_components.items()
    )


def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    """
    Build subgraph from triples
    :param num_nodes: number of nodes
    :param num_rels: number of relations
    :param triples: array-like with columns [src, rel, dst]
    :param use_cuda: whether to use CUDA
    :param gpu: GPU device ID
    :return: PyTorch Geometric Data object
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import degree

    triples = np.asarray(triples)
    if triples.shape[-1] > 3:
        triples = triples[:, :3]
    triples = triples.astype(np.int64, copy=False)
    triples = triples.reshape(-1, 3)

    src, rel, dst = triples.transpose()

    src = np.concatenate((src, dst))
    dst = np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    min_len = min(len(src), len(dst), len(rel))
    if min_len == 0:
        raise ValueError("Empty triples after preprocessing.")
    if not (len(src) == len(dst) == len(rel)):
        src = src[:min_len]
        dst = dst[:min_len]
        rel = rel[:min_len]

    src_t = torch.from_numpy(src).long()
    dst_t = torch.from_numpy(dst).long()
    rel_t = torch.from_numpy(rel).long()

    edge_index = torch.stack([src_t, dst_t], dim=0)

    in_deg = degree(dst_t, num_nodes=num_nodes).float()
    in_deg[in_deg == 0] = 1.0
    norm = 1.0 / in_deg

    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)

    data = Data()
    data.num_nodes = num_nodes
    data.edge_index = edge_index
    data.id = node_id
    data.node_norm = norm.view(-1, 1)
    edge_norm = data.node_norm[dst_t] * data.node_norm[src_t]
    data.edge_norm = edge_norm
    data.edge_type = rel_t

    data.uniq_r = torch.from_numpy(uniq_r).long()
    data.r_len = torch.from_numpy(np.array(r_len)).long()
    data.r_to_e = torch.from_numpy(np.array(r_to_e)).long()

    if use_cuda:
        device = torch.device(f"cuda:{gpu}" if isinstance(gpu, int) else "cuda")
        data = data.to(device)
    return data


def _read_triplets_as_list(filename, entity_dict, relation_dict, load_time):
    l = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            if len(processed_line) < 3:
                continue
            s = int(processed_line[0])
            r = int(processed_line[1])
            o = int(processed_line[2])
            if load_time:
                if len(processed_line) >= 4:
                    st = int(processed_line[3])
                    l.append([s, r, o, st])
                else:
                    l.append([s, r, o])
            else:
                l.append([s, r, o])
    return l


def _read_dictionary(filename):
    d = {}
    try:
        f = open(filename, 'r', encoding='utf-8', errors='ignore')
    except TypeError:
        f = open(filename, 'r')
    with f:
        for line in f:
            line = line.strip().split('\t')
            if len(line) < 2:
                continue
            d[int(line[1])] = line[0]
    return d


class RGCNLinkDataset(object):
    """RGCN link prediction dataset"""

    def __init__(self, name, dir=None):
        self.name = name
        if dir:
            self.dir = dir
            self.dir = os.path.join(self.dir, self.name)
        else:
            raise ValueError("dir parameter is required")
        print(self.dir)

    def load(self, load_time=True):
        entity_path = os.path.join(self.dir, 'entity2id.txt')
        relation_path = os.path.join(self.dir, 'relation2id.txt')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.array(_read_triplets_as_list(train_path, entity_dict, relation_dict, load_time))
        self.valid = np.array(_read_triplets_as_list(valid_path, entity_dict, relation_dict, load_time))
        self.test = np.array(_read_triplets_as_list(test_path, entity_dict, relation_dict, load_time))
        self.num_nodes = len(entity_dict)
        print("# Sanity Check:  entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        print("# Sanity Check:  relations: {}".format(self.num_rels))
        print("# Sanity Check:  edges: {}".format(len(self.train)))


def load_from_local(dir, dataset):
    """Load dataset from local directory """
    data = RGCNLinkDataset(dataset, dir)
    data.load()
    return data


def resolve_data_base_dir(dataset):
    base_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir) == 'code' else base_dir

    candidate_bases = [
        os.path.join(base_dir, '..', 'adhub', 'ICEWS18', '0.1', 'data'),  
        os.path.join(base_dir, '..', 'data'),  
        os.path.join(os.getcwd(), 'data'),
        os.path.join(base_dir, 'data'),
        os.path.join(project_root, 'data'),
    ]

    seen = set()
    unique_candidates = []
    for base in candidate_bases:
        if base not in seen:
            seen.add(base)
            unique_candidates.append(base)

    for base in unique_candidates:
        entity_path = os.path.join(base, dataset, 'entity2id.txt')
        if os.path.isfile(entity_path):
            logging.info(f"Using dataset path: {entity_path}")
            return base
        
        direct_entity_path = os.path.join(base, 'entity2id.txt')
        if os.path.isfile(direct_entity_path):
            logging.info(f"Using dataset path: {direct_entity_path}")
            return base

    raise FileNotFoundError(
        f"Could not find local dataset files for {dataset}. "
        f"Checked local bases: {unique_candidates}. "
        f"For server: ensure ../adhub/ICEWS14/0.1/data/{dataset}/ exists with entity2id.txt"
    )


class PrintToLog:
    def write(self, message):
        if message != '\n':
            logging.info(message)

    def flush(self):
        pass


def setup_logging(log_file):
    logging.basicConfig(filename=log_file,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


sys.stdout = PrintToLog()


def create_timestamped_dir(base_dir, args):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(base_dir, timestamp)
    results_dir = results_dir + f"_dataset_{args.dataset}_history_len_{args.history_len}_train_history_len_{args.train_history_len}_max_length_{args.max_length}_hidden_dim_{args.hidden_dim}_num_layers_{args.num_layers}_num_heads_{args.num_heads}_num_codes_{args.num_code}_ratio_{args.split_ratio}_tips_{args.tips}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def re_spilt(dataset, split_ratio=[0.5, 0.2, 0.3]):
    all_triples = np.concatenate([dataset.train, dataset.valid, dataset.test], axis=0)
    times = np.unique(all_triples[:, 3])
    train_time = times[:int(len(times) * split_ratio[0])]
    valid_time = times[int(len(times) * split_ratio[0]):int(len(times) * (split_ratio[0] + split_ratio[1]))]
    test_time = times[int(len(times) * (split_ratio[0] + split_ratio[1])):]
    train_triples = all_triples[all_triples[:, 3] <= train_time[-1], :]
    val_triples = all_triples[(all_triples[:, 3] > train_time[-1]) & (all_triples[:, 3] <= valid_time[-1])]
    test_triples = all_triples[(all_triples[:, 3] > valid_time[-1]) & (all_triples[:, 3] <= test_time[-1])]
    dataset.train = train_triples
    dataset.valid = val_triples
    dataset.test = test_triples
    total_triples = len(all_triples)
    train_count = len(train_triples)
    val_count = len(val_triples)
    test_count = len(test_triples)
    logging.info(f"[Data Split] Total triples: {total_triples}")
    logging.info(f"[Data Split] Train: {train_count} ({train_count/total_triples:.1%}), "
                 f"Valid: {val_count} ({val_count/total_triples:.1%}), "
                 f"Test: {test_count} ({test_count/total_triples:.1%})")
    logging.info(f"[Data Split] Time periods - Train: {len(train_time)}, Valid: {len(valid_time)}, Test: {len(test_time)}")
    return dataset


class TKGDataset(Dataset):
    def __init__(self, data_dict, all_triples):
        self.data_dict = data_dict
        self.triples = all_triples
        self.times = np.unique(all_triples[:, 3])
        self.num_relations = np.unique(all_triples[:, 1]).shape[0]
        self.num_entities = np.unique(all_triples[:, [0, 2]]).shape[0]

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        time = self.times[idx]
        triples_at_time = self.triples[self.triples[:, 3] == time]
        return torch.tensor(triples_at_time)


def select_trace_events_by_relation(query_relation, chain_relation, relation_embedding, topk=30):
    query_relation_embedding = relation_embedding[query_relation]
    chain_relation_embedding = relation_embedding[chain_relation]
    query_relation_embedding = query_relation_embedding.unsqueeze(0)
    similarity = torch.matmul(query_relation_embedding, chain_relation_embedding.T)
    _, topk_index = torch.topk(similarity, topk)
    return topk_index


def encode_interaction_trace(head, interaction_trace, embedding_dict, model, device='cuda'):
    entity_embedding, relation_embedding, cls_embedding, empty_embedding = embedding_dict['entity_embedding'],\
        embedding_dict['relation_embedding'], embedding_dict['cls_embedding'], embedding_dict['missing_embedding']
    B, N, M = interaction_trace.shape[0], interaction_trace.shape[1], interaction_trace.shape[2]
    interaction_trace = interaction_trace.to(torch.int64)
    interaction_trace = sort_by_last_dim_with_neg1_last(interaction_trace)
    time_projection = model.time_projection
    entity_proj = model.entity_down_proj
    relation_proj = model.relation_down_proj
    trace_mask = interaction_trace[:, :, 0] == -1
    valid_trace_events = interaction_trace[~trace_mask]
    s, r, o, t = valid_trace_events[:, 0], valid_trace_events[:, 1], valid_trace_events[:, 2], valid_trace_events[:, 3]
    trace_embedding = torch.zeros(B, N, M, int(relation_embedding.shape[1] / M),
                                  device=relation_embedding.device)
    trace_embedding[:, :, :, :] = empty_embedding[:, int(relation_embedding.shape[1] / M)]
    valid_event_embedding = torch.zeros(len(valid_trace_events), M, int(relation_embedding.shape[1] / 4),
                                        device=relation_embedding.device)
    valid_event_embedding[:, 0] = entity_proj(entity_embedding[s])
    valid_event_embedding[:, 1] = relation_proj(relation_embedding[r])
    valid_event_embedding[:, 2] = entity_proj(entity_embedding[o])
    valid_event_embedding[:, 3] = time_projection(t.unsqueeze(-1).float())
    unmasked_indices = torch.nonzero(~trace_mask, as_tuple=False)  
    i_idx, j_idx = unmasked_indices[:, 0], unmasked_indices[:, 1]
    trace_embedding[:, 0, 0] = entity_proj(entity_embedding[head])
    trace_embedding[i_idx, j_idx] = valid_event_embedding
    trace_embedding = trace_embedding.view(B, -1, relation_embedding.shape[1])
    return trace_embedding, trace_mask


def build_history_graphs_fast(train_list, current_time_idx, num_nodes, num_rels, history_len, device, use_cuda=True):
    if current_time_idx == 0:
        return [None] * history_len

    if current_time_idx - history_len < 0:
        input_list = train_list[0: current_time_idx]
    else:
        input_list = train_list[current_time_idx - history_len: current_time_idx]

    history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, device) for snap in input_list]

    return history_glist


def build_interaction_trace_batch(ground_truth, data, embedding_dict, model, trace_length=30):
    length = len(ground_truth)
    interaction_trace_all = np.zeros((length, trace_length, 4)) - 1
    entity_embedding, relation_embedding, cls_embedding, missing_embedding = embedding_dict['entity_embedding'],\
        embedding_dict['relation_embedding'], embedding_dict['cls_embedding'], embedding_dict['missing_embedding']
    for idx in range(length):
        interaction_trace = data[idx].get('context_window', data[idx].get('one_hop_chain'))
        if len(interaction_trace) > trace_length:
            query_relation = ground_truth[idx, 1]
            chain_relation = interaction_trace[:, 1]
            topk_index = select_trace_events_by_relation(query_relation, chain_relation, relation_embedding,
                                                         topk=trace_length)
            topk_idx = topk_index.squeeze()
            interaction_trace = interaction_trace[topk_idx.cpu().numpy()] if isinstance(interaction_trace, np.ndarray) else\
                interaction_trace[topk_idx]
        interaction_trace_all[idx, :len(interaction_trace), :4] = interaction_trace
        interaction_trace_all[idx, :len(interaction_trace), 3] = ground_truth[idx, 3] - interaction_trace_all[idx,
                                                                                     :len(interaction_trace), 3]
    head = ground_truth[:, 0]

    device = next(model.parameters()).device
    trace_embedding, trace_mask = encode_interaction_trace(head, torch.tensor(interaction_trace_all, device=device),
                                                           embedding_dict, model, device=device)
    return trace_embedding, trace_mask, interaction_trace_all


def get_init_embedding(entity_embedding_path, n_entity, n_relation, hidden_dim, device='cuda', word_embedding=True):
    gamma = 6.0
    epsilon = 1.0
    embedding_range = nn.Parameter(
        torch.Tensor([(gamma + epsilon) / hidden_dim]),
        requires_grad=False
    )
    if word_embedding:
        if os.path.exists(entity_embedding_path):
            entity_embedding = torch.tensor(np.load(entity_embedding_path), dtype=torch.float).to(device)
            logging.info(
                f"[Model Init] Loaded BERT entity embeddings from {entity_embedding_path}, shape={entity_embedding.shape}")
        else:
            logging.warning(
                f"[Model Init] BERT entity embedding file not found at {entity_embedding_path}, using random initialization")
            entity_embedding = nn.Parameter(torch.zeros(n_entity, hidden_dim, device=device))
            nn.init.uniform_(
                tensor=entity_embedding,
                a=-embedding_range.item(),
                b=embedding_range.item()
            )
    else:
        entity_embedding = nn.Parameter(torch.zeros(n_entity, hidden_dim, device=device))
        nn.init.uniform_(
            tensor=entity_embedding,
            a=-embedding_range.item(),
            b=embedding_range.item()
        )
    relation_embedding = nn.Parameter(torch.zeros(n_relation * 2, hidden_dim, device=device))
    nn.init.uniform_(
        tensor=relation_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    cls_embedding = nn.Parameter(torch.zeros(4, hidden_dim, device=device))
    missing_embedding = nn.Parameter(torch.zeros((1, hidden_dim), device=device))
    nn.init.uniform_(
        tensor=cls_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    nn.init.uniform_(
        tensor=missing_embedding,
        a=-embedding_range.item(),
        b=embedding_range.item()
    )
    embedding_dict = {}
    embedding_dict['entity_embedding'] = entity_embedding
    embedding_dict['relation_embedding'] = relation_embedding
    embedding_dict['cls_embedding'] = cls_embedding
    embedding_dict['missing_embedding'] = missing_embedding
    return embedding_dict


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def parse_args():
    parser = argparse.ArgumentParser(description="Temporal Knowledge Graph Embedding")
    parser.add_argument("--dataset", type=str, default="ICEWS14", 
                        choices=["ICEWS14", "ICEWS18", "ICEWS05-15", "GDELT"],
                        help="dataset name. Supported: ICEWS14, ICEWS18, ICEWS05-15, GDELT")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--history_len", type=int, default=14, help="train history length")
    parser.add_argument("--train_history_len", type=int, default=7,
                        help="history length for the transductive stream")
    parser.add_argument("--max_length", type=int, default=30, help="max length of thechain")
    parser.add_argument("--hidden_dim", type=int, default=768, help="hidden dimension")
    parser.add_argument("--trans_hidden_dim", type=int, default=256,
                        help="hidden dimension of the transductive stream")
    parser.add_argument("--trans_dropout", type=float, default=0.2,
                        help="dropout of the transductive stream")
    parser.add_argument("--trans_num_bases", type=int, default=128,
                        help="number of basis matrices for the structural branch")
    parser.add_argument("--trans_num_basis", type=int, default=128,
                        help="basis size for relation-specific transformations")
    parser.add_argument("--trans_score_weight", type=float, default=0.9,
                        help="decoder blending weight used by the transductive stream")
    parser.add_argument("--contrastive_temperature", type=float, default=0.03,
                        help="temperature for temporal contrastive optimization")
    parser.add_argument("--trans_input_dropout", type=float, default=0.2,
                        help="transductive decoder input dropout")
    parser.add_argument("--trans_hidden_dropout", type=float, default=0.2,
                        help="transductive decoder hidden dropout")
    parser.add_argument("--trans_feat_dropout", type=float, default=0.2,
                        help="transductive decoder feature dropout")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--word_embedding", action='store_false', default=True, help="word embedding")
    parser.add_argument("--word_embedding_path", type=str, default='data', help="word embedding path")
    parser.add_argument("--word_embedding_dim", type=int, default=768, help="word embedding dimension")
    parser.add_argument("--residual", type=bool, default=True, help="residual")
    parser.add_argument("--result_dir", type=str, default=None, help="result dir (default: ../results for server)")
    parser.add_argument("--layer_norm", action='store_false', default=True, help="layer norm")
    parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
    parser.add_argument("--tips", type=str, default='None', help="use tips")
    parser.add_argument("--patience", type=int, default=3, help="patience for early stopping")
    parser.add_argument("--seed", type=int, nargs='+', default=[42], help="list of random seeds")
    parser.add_argument("--num_code", type=int, default=50, help="number of clusters for clustering")
    parser.add_argument("--split_ratio", type=str, default="50_20_30", 
                        choices=["20_40_40", "30_30_40", "40_30_30", "50_20_30", "60_20_20", "70_10_20", "80_10_10"],
                        help="train/val/test split ratio. Format: train_val_test (e.g., '50_20_30' means 50%% train, 20%% val, 30%% test)")
    parser.add_argument("--no_trans", action='store_true', default=False, help="disable the transductive stream")
    parser.add_argument("--add_static_graph", action='store_true', default=True,
                        help="use lexical/static relational context")
    return parser.parse_args()


def main(args):
    if args.result_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.result_dir = os.path.join(script_dir, 'results')
        os.makedirs(args.result_dir, exist_ok=True)
    
    results_dir = create_timestamped_dir(args.result_dir, args)
    setup_logging(os.path.join(results_dir, 'log.txt'))
    data_base_dir = resolve_data_base_dir(args.dataset)
    args.word_embedding_path = os.path.join(data_base_dir, args.dataset, f'{args.dataset}_Bert_Entity_Embedding.npy')
    print(args)
    gpu_id = args.gpu
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    print("loading graph data")
    data = load_from_local(data_base_dir, args.dataset)
    ratio_parts = args.split_ratio.split('_')
    if len(ratio_parts) != 3:
        raise ValueError(f"Invalid split_ratio format: {args.split_ratio}. Expected format: 'train_val_test' (e.g., '50_20_30')")
    split_ratio = [int(ratio_parts[0]) / 100.0, int(ratio_parts[1]) / 100.0, int(ratio_parts[2]) / 100.0]
    ratio_sum = sum(split_ratio)
    if abs(ratio_sum - 1.0) > 0.01:
        raise ValueError(f"Split ratios must sum to 100%, but got {ratio_sum*100:.1f}%")
    logging.info(f"Using split ratio: train={split_ratio[0]:.1%}, val={split_ratio[1]:.1%}, test={split_ratio[2]:.1%}")
    data = re_spilt(data, split_ratio)
    known_entity = set(data.train[:, 0].tolist() + data.train[:, 2].tolist())
    known_entity_index = torch.tensor(list(known_entity), dtype=torch.long)
    print("Number of known entities: ", len(known_entity))

    known_entity_train_val = set(
        data.train[:, 0].tolist() + data.train[:, 2].tolist() + data.valid[:, 0].tolist() + data.valid[:, 2].tolist())
    known_entity_train_val_index = torch.tensor(list(known_entity_train_val), dtype=torch.long)
    train_triple = data.train
    valid_triple = data.valid
    test_triple = data.test
    all_triple = np.concatenate([train_triple, valid_triple, test_triple], axis=0)
    times = np.unique(all_triple[:, 3])
    train_time = times[:int(len(times) * split_ratio[0])]
    valid_time = times[int(len(times) * split_ratio[0]):int(len(times) * (split_ratio[0] + split_ratio[1]))]
    test_time = times[int(len(times) * (split_ratio[0] + split_ratio[1])):]
    test_time = test_time[:-1]
    logging.info(f"[Time Split] Total time periods: {len(times)}")
    logging.info(f"[Time Split] Train periods: {len(train_time)} ({len(train_time)/len(times):.1%}), "
                 f"Valid periods: {len(valid_time)} ({len(valid_time)/len(times):.1%}), "
                 f"Test periods: {len(test_time)} ({len(test_time)/len(times):.1%})")

    all_entities = np.concatenate([all_triple[:, 0], all_triple[:, 2]])
    all_entities = np.unique(all_entities)
    all_relations = np.unique(all_triple[:, 1])

    all_triple_original = all_triple.copy()

    original_relation_count = len(all_relations)
    all_triple_inverse = all_triple[:, [2, 1, 0, 3]]
    all_triple_inverse[:, 1] += original_relation_count
    all_triple = np.concatenate([all_triple, all_triple_inverse], axis=0)

    if args.add_static_graph and not args.no_trans:
        try:
            static_graph_path = os.path.join(data_base_dir, args.dataset, 'e-w-graph.txt')
            if os.path.exists(static_graph_path):
                static_triples = np.array(_read_triplets_as_list(static_graph_path, {}, {}, load_time=False))
                num_static_rels = len(np.unique(static_triples[:, 1]))
                num_words = len(np.unique(static_triples[:, 2]))
                static_triples[:, 2] = static_triples[:, 2] + data.num_nodes
                static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().to(device)\
                    if torch.cuda.is_available() else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1,
                                                                                                                   1).long()
                static_graph = None
                print(f"[Static Graph] Loaded static graph from {static_graph_path}")
                print(f"[Static Graph] num_static_rels={num_static_rels}, num_words={num_words}")
            else:
                print(
                    f"[Static Graph] Warning: Static graph file not found at {static_graph_path}, continuing without static graph")
                num_static_rels, num_words, static_triples, static_node_id, static_graph = 0, 0, [], None, None
        except Exception as e:
            print(f"[Static Graph] Warning: Failed to load static graph: {e}, continuing without static graph")
            num_static_rels, num_words, static_triples, static_node_id, static_graph = 0, 0, [], None, None
    else:
        num_static_rels, num_words, static_triples, static_node_id, static_graph = 0, 0, [], None, None

    entity_history = {}
    for entity in all_entities:
        triple_with_entity = all_triple[all_triple[:, 0] == entity]
        triple_with_entity_sort_by_time = triple_with_entity[np.argsort(triple_with_entity[:, 3])]
        entity_history[entity] = triple_with_entity_sort_by_time

    dataset = TKGDataset(entity_history, all_triple)

    file_path = os.path.join(data_base_dir, args.dataset, f'{args.dataset}_T_{args.history_len}.pkl')
    hisotry_dataset = load_pickle_auto(file_path)

    train_list = []
    for time in train_time:
        triples_at_time = all_triple_original[all_triple_original[:, 3] == time].copy()
        train_list.append(triples_at_time)

    valid_list = []
    for time in valid_time:
        triples_at_time = all_triple_original[all_triple_original[:, 3] == time].copy()
        valid_list.append(triples_at_time)

    test_list = []
    for time in test_time:
        triples_at_time = all_triple_original[all_triple_original[:, 3] == time].copy()
        test_list.append(triples_at_time)

    history_cache = {'train': {}, 'valid': {}, 'test': {}}

    best_result_dict = {}
    for seed in args.seed:
        set_random_seed(seed)
        print(f"Running with seed {seed}")
        embedding_dict = get_init_embedding(args.word_embedding_path, data.num_nodes, data.num_rels, args.hidden_dim,
                                            device=device)
        original_num_rels = len(all_relations)
        model = DoradoModel(data.num_nodes, original_num_rels, num_heads=args.num_heads, entity_dim=args.hidden_dim,
                            relation_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout,
                            word_embedding=args.word_embedding, word_embedding_path=args.word_embedding_path,
                            layer_norm=args.layer_norm, word_embedding_dim=args.word_embedding_dim, num_code=args.num_code,
                            dataset_name=args.dataset, trans_hidden_dim=args.trans_hidden_dim,
                            trans_dropout=args.trans_dropout, trans_num_bases=args.trans_num_bases,
                            trans_num_basis=args.trans_num_basis, trans_score_weight=args.trans_score_weight,
                            contrastive_temperature=args.contrastive_temperature, trans_input_dropout=args.trans_input_dropout,
                            trans_hidden_dropout=args.trans_hidden_dropout, trans_feat_dropout=args.trans_feat_dropout,
                            use_static_background=args.add_static_graph, num_static_relations=num_static_rels,
                            num_static_words=num_words).to(device)
        if args.no_trans:
            model.use_transductive_stream = False
        if hasattr(model, 'trans_fast_mode'):
            model.trans_fast_mode = True
        if args.add_static_graph and not args.no_trans and static_graph is None and len(static_triples) > 0:
            static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda=True,
                                           gpu=gpu_id)
            print(
                f"[Static Graph] Built static graph with {len(static_triples)} edges, num_nodes={len(static_node_id)}")

        all_params = list(model.parameters()) + [embedding_dict['relation_embedding'], embedding_dict['cls_embedding'],
                                                 embedding_dict['missing_embedding']]
        optimizer = torch.optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)
        best_val_loss = 100000
        patience = 0
        max_patience = args.patience
        model_name = f"model_dataset_{args.dataset}_history_len_{args.history_len}_max_length_{args.max_length}_hidden_dim_{args.hidden_dim}_num_layers_{args.num_layers}_num_heads_{args.num_heads}_num_codes_{args.num_code}_tips_{args.tips}_seed_{seed}.pth"
        best_val_mrr = 0
        test_emerging_triple_index = None
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            train_loss_list = []
            train_loss_components = init_loss_components()
            unvalid_ratio = 0

            for year in tqdm(range(len(train_time))):
                if year == 0: continue

                data_for_year = dataset[year]
                history_data = hisotry_dataset[year]

                if not args.no_trans:
                    history_glist = history_cache['train'].get(year)
                    if history_glist is None:
                        history_glist = build_history_graphs_fast(
                            train_list, year, data.num_nodes, data.num_rels,
                            args.train_history_len, gpu_id, use_cuda=True
                        )
                        history_cache['train'][year] = history_glist
                else:
                    history_glist = None

                for batch_id in range(2):
                    if not args.no_trans:
                        batch_history_glist = history_glist
                    else:
                        batch_history_glist = None

                    start_id = int(batch_id * len(data_for_year) / 2)
                    end_id = int((batch_id + 1) * len(data_for_year) / 2)
                    triples = data_for_year[start_id:end_id]
                    batch_history_data = [history_data[i] for i in range(start_id, end_id)]
                    trace_embedding, trace_mask, _ = build_interaction_trace_batch(triples, batch_history_data,
                                                                                   embedding_dict, model,
                                                                                   trace_length=args.max_length)

                    triples = triples.to(device)
                    score, loss, task_loss, ortho_loss, temporal_contrastive_loss, distill_loss = model(
                        triples, trace_embedding, trace_mask,
                        embedding_dict,
                        history_glist=batch_history_glist,
                        static_graph=static_graph, use_cuda=True,
                        T_idx=year)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    loss_val = loss.item()
                    train_loss += loss_val
                    train_loss_list.append(loss_val)
                    record_loss_components(
                        train_loss_components, task_loss, ortho_loss,
                        temporal_contrastive_loss, distill_loss)
                    del score, loss, task_loss, ortho_loss, temporal_contrastive_loss, distill_loss
                    del trace_embedding, trace_mask

            del history_glist, data_for_year, history_data

            unvalid_ratio /= len(train_time)

            avg_loss = np.mean(train_loss_list) if len(train_loss_list) > 0 else 0.0
            print(f"Epoch: {epoch:04d} | Ave Loss: {avg_loss:.4f}")
            print(f"Train Loss Components | {format_loss_components(train_loss_components)}")

            model.eval()
            valid_loss = 0

            valid_loss_list = []
            rank_list = []
            unvalid_ratio = 0
            with torch.no_grad():
                for year in range(len(train_time), len(train_time) + len(valid_time)):
                    data_for_year = dataset[year]
                    history_data = hisotry_dataset[year]

                    if not args.no_trans:
                        valid_year_idx = year - len(train_time)
                        cached = history_cache['valid'].get(valid_year_idx)
                        if cached is None:
                            combined_list = train_list + valid_list[:valid_year_idx]
                            cached = build_history_graphs_fast(
                                combined_list, len(combined_list), data.num_nodes, data.num_rels,
                                args.train_history_len, gpu_id, use_cuda=True
                            )
                            history_cache['valid'][valid_year_idx] = cached
                        history_glist = cached
                    else:
                        history_glist = None

                    trace_embedding_all, trace_mask_all, _ = build_interaction_trace_batch(data_for_year,
                                                                                  history_data,
                                                                                  embedding_dict,
                                                                                  model,
                                                                                  trace_length=args.max_length)

                    for batch_id in range(2):
                        start_id = int(batch_id * len(data_for_year) / 2)
                        end_id = int((batch_id + 1) * len(data_for_year) / 2)
                        triples = data_for_year[start_id:end_id]
                        trace_embedding = trace_embedding_all[start_id:end_id]
                        trace_mask = trace_mask_all[start_id:end_id]
                        triples = triples.to(device)
                        score, loss, task_loss, ortho_loss, temporal_contrastive_loss, distill_loss = model(
                            triples, trace_embedding, trace_mask,
                            embedding_dict, history_glist=history_glist,
                            static_graph=static_graph, use_cuda=True,
                            T_idx=year)
                        valid_loss += loss.item()
                        valid_loss_list.append(loss.item())
                        rank = utils.get_rank(score, triples[:, 2])
                        rank_list.append(rank.cpu())

                unvalid_ratio /= len(valid_time)
                mrr, hit1, hit3, hit10 = utils.get_metric(rank_list)
                print(f"Epoch: {epoch}, Valid Loss: {valid_loss / len(valid_loss_list):.4f}")

            if mrr > best_val_mrr:
                best_val_mrr = mrr
                test_loss = 0
                test_loss_list = []
                rank_list = []
                test_triples = []
                unvalid_ratio = 0
                with torch.no_grad():
                    for year in range(len(train_time) + len(valid_time),
                                      len(train_time) + len(valid_time) + len(test_time) - 1):
                        data_for_year = dataset[year]
                        history_data = hisotry_dataset[year]

                        if not args.no_trans:
                            test_year_idx = year - len(train_time) - len(valid_time)
                            cached = history_cache['test'].get(test_year_idx)
                            if cached is None:
                                combined_list = train_list + valid_list + test_list[:test_year_idx]
                                cached = build_history_graphs_fast(
                                    combined_list, len(combined_list), data.num_nodes, data.num_rels,
                                    args.train_history_len, gpu_id, use_cuda=True
                                )
                                history_cache['test'][test_year_idx] = cached
                            history_glist = cached
                        else:
                            history_glist = None

                        batch_triples = np.zeros((0, 4))
                        trace_embedding_all, trace_mask_all, _ = build_interaction_trace_batch(
                            data_for_year, history_data, embedding_dict, model, trace_length=args.max_length)

                        for batch_id in range(2):
                            start_id = int(batch_id * len(data_for_year) / 2)
                            end_id = int((batch_id + 1) * len(data_for_year) / 2)
                            triples = data_for_year[start_id:end_id]
                            trace_embedding = trace_embedding_all[start_id:end_id]
                            trace_mask = trace_mask_all[start_id:end_id]
                            triples = triples.to(device)
                            score, loss, task_loss, ortho_loss, temporal_contrastive_loss, distill_loss = model(
                                triples, trace_embedding, trace_mask,
                                embedding_dict,
                                history_glist=history_glist,
                                static_graph=static_graph, use_cuda=True,
                                T_idx=year)
                            test_loss += loss.item()
                            test_loss_list.append(loss.item())
                            rank = utils.get_rank(score, triples[:, 2])
                            rank_list.append(rank.cpu())
                            batch_triples = np.concatenate((batch_triples, triples.cpu().numpy()), axis=0)
                        test_triples.append(batch_triples)
                    unvalid_ratio /= len(test_time)
                    if test_emerging_triple_index is None:
                        test_emerging_triple_index = utils.get_emerging_index(test_triples,
                                                                              known_entity_train_val_index)
                    mrr, hit1, hit3, hit10 = utils.get_metric(rank_list)
                    emerging_mrr_test, emerging_hit1, emerging_hit3, emerging_hit10 = utils.get_metric_emerging_both(
                        rank_list, test_triples, test_emerging_triple_index)
                    known_mrr_test, known_hit1, known_hit3, known_hit10 = utils.get_metric_known_both(
                        rank_list, test_triples, test_emerging_triple_index)

                    print(f"Epoch: {epoch}, Test Loss: {test_loss / len(test_loss_list):.4f}")
                    print(f"MRR: {mrr:.4f}, Hit1: {hit1:.4f}, Hit3: {hit3:.4f}, Hit10: {hit10:.4f}")
                    print(
                        f"Emerging_MRR: {emerging_mrr_test:.4f}, Emerging_Hit1: {emerging_hit1:.4f}, Emerging_Hit3: {emerging_hit3:.4f}, Emerging_Hit10: {emerging_hit10:.4f}")
                    print(
                        f"Known_MRR: {known_mrr_test:.4f}, Known_Hit1: {known_hit1:.4f}, Known_Hit3: {known_hit3:.4f}, Known_Hit10: {known_hit10:.4f}")
                    best_test_mertic = f"Emerging_MRR: {emerging_mrr_test:.4f}, Emerging_Hit1: {emerging_hit1:.4f}, Emerging_Hit3: {emerging_hit3:.4f}, Emerging_Hit10: {emerging_hit10:.4f}"
                    best_result = {'all_mrr': mrr, 'all_hit1': hit1, 'all_hit3': hit3, 'all_hit10': hit10,
                                   'emerging_mrr': emerging_mrr_test, 'emerging_hit1': emerging_hit1,
                                   'emerging_hit3': emerging_hit3, 'emerging_hit10': emerging_hit10,
                                   'known_mrr': known_mrr_test, 'known_hit1': known_hit1,
                                   'known_hit3': known_hit3, 'known_hit10': known_hit10}
                patience = 0
                os.makedirs(results_dir, exist_ok=True)

                model_name_short = f"model_{args.dataset}_seed_{seed}.pth"
                embedding_name_short = f"embedding_{args.dataset}_seed_{seed}.pth"

                model_path = os.path.join(results_dir, model_name_short)
                embedding_path = os.path.join(results_dir, embedding_name_short)

                full_model_path = os.path.abspath(model_path)
                if len(full_model_path) > 250:
                    print(
                        f"Warning: Model path is very long ({len(full_model_path)} chars). Consider using shorter paths.")

                try:
                    torch.save(model.state_dict(), model_path)
                    torch.save(embedding_dict, embedding_path)
                    print(f"Model saved to: {model_path}")
                    print(f"Embedding saved to: {embedding_path}")
                except Exception as e:
                    print(f"Error saving model: {e}")
                    print(f"Model path: {model_path}")
                    print(f"Full path length: {len(os.path.abspath(model_path))}")
                    raise
            else:
                patience += 1
            if patience > max_patience:
                print(f"Early stopping at epoch {epoch}, best validation MRR: {best_val_mrr:.4f}")
                print(f"Best test metrics: {best_test_mertic}")
                best_result_dict[seed] = best_result
                break
            if epoch == args.epochs - 1:
                print(f"Reached maximum epochs {args.epochs}, best validation MRR: {best_val_mrr:.4f}")
                print(f"Best test metrics: {best_test_mertic}")
                best_result_dict[seed] = best_result

    print("Best results for each seed:")
    for seed, result in best_result_dict.items():
        print(
            f"Seed {seed}: MRR: {result['all_mrr']:.4f}, Hit1: {result['all_hit1']:.4f}, Hit3: {result['all_hit3']:.4f}, Hit10: {result['all_hit10']:.4f}, "
            f"Emerging_MRR: {result['emerging_mrr']:.4f}, Emerging_Hit1: {result['emerging_hit1']:.4f}, Emerging_Hit3: {result['emerging_hit3']:.4f}, Emerging_Hit10: {result['emerging_hit10']:.4f}, "
            f"Known_MRR: {result['known_mrr']:.4f}, Known_Hit1: {result['known_hit1']:.4f}, Known_Hit3: {result['known_hit3']:.4f}, Known_Hit10: {result['known_hit10']:.4f}")
    print("Average results:")
    avg_result = {}
    for key in result.keys():
        avg_result[key] = np.mean([result[key] for result in best_result_dict.values()])
    print(
        f"Average MRR: {avg_result['all_mrr']:.4f}, Average Hit1: {avg_result['all_hit1']:.4f}, Average Hit3: {avg_result['all_hit3']:.4f}, Average Hit10: {avg_result['all_hit10']:.4f}, "
        f"Average Emerging_MRR: {avg_result['emerging_mrr']:.4f}, Average Emerging_Hit1: {avg_result['emerging_hit1']:.4f}, Average Emerging_Hit3: {avg_result['emerging_hit3']:.4f}, Average Emerging_Hit10: {avg_result['emerging_hit10']:.4f}, "
        f"Average Known_MRR: {avg_result['known_mrr']:.4f}, Average Known_Hit1: {avg_result['known_hit1']:.4f}, Average Known_Hit3: {avg_result['known_hit3']:.4f}, Average Known_Hit10: {avg_result['known_hit10']:.4f}")

if __name__ == "__main__":
    import sys

    original_stdout = sys.__stdout__

    args = parse_args()

    split_ratio_values = ["5_45_50", "10_40_50", "20_40_40", "30_30_40", "40_30_30", "50_20_30", "60_20_20", "70_10_20", "80_10_10"]

    original_stdout.write("=" * 80 + "\n")
    original_stdout.write(f"Batch running split_ratio values: {split_ratio_values}\n")
    original_stdout.write(f"Dataset: {args.dataset}\n")
    original_stdout.write("=" * 80 + "\n")
    original_stdout.flush()

    all_results = {}

    for idx, split_ratio in enumerate(split_ratio_values, 1):
        original_stdout.write("\n" + "=" * 80 + "\n")
        original_stdout.write(f"Run {idx}/{len(split_ratio_values)}: split_ratio = {split_ratio}\n")
        original_stdout.write("=" * 80 + "\n")
        original_stdout.flush()

        args.split_ratio = split_ratio

        try:
            main(args)
            all_results[split_ratio] = "done"
            original_stdout.write(f"\n[OK] split_ratio = {split_ratio} finished\n")
            original_stdout.flush()
        except Exception as e:
            all_results[split_ratio] = f"error: {str(e)}"
            original_stdout.write(f"\n[FAIL] split_ratio = {split_ratio} failed: {str(e)}\n")
            original_stdout.flush()
            import traceback
            traceback.print_exc()

    original_stdout.write("\n" + "=" * 80 + "\n")
    original_stdout.write("split_ratio batch run summary\n")
    original_stdout.write("=" * 80 + "\n")
    for split_ratio, status in all_results.items():
        original_stdout.write(f"split_ratio = {split_ratio}: {status}\n")
    original_stdout.write("=" * 80 + "\n")
    original_stdout.flush()
