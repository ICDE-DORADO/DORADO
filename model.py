import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_geometric.nn.conv import MessagePassing
import math
from torch_geometric.utils import scatter
import sys
import os
from collections import defaultdict


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight,
                                    gain=nn.init.calculate_gain('relu'))

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    def forward(self, x, edge_index, edge_type, edge_norm, prev_h=[]):
        if self.self_loop:
            loop_message = torch.mm(x, self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)

        # message passing implemented in subclasses
        node_repr = self.propagate_layer(x, edge_index, edge_type, edge_norm)
        if self.bias is not None:
            node_repr = node_repr + self.bias
        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
        return node_repr

    def propagate_layer(self, x, edge_index, edge_type, edge_norm):
        raise NotImplementedError


class RGCNBlockLayer(RGCNLayer, MessagePassing):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        RGCNLayer.__init__(self, in_feat, out_feat, bias,
                           activation, self_loop=self_loop, skip_connect=skip_connect,
                           dropout=dropout)
        MessagePassing.__init__(self, aggr='add')
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def propagate_layer(self, x, edge_index, edge_type, edge_norm):
        weight = self.weight.index_select(0, edge_type).view(
            -1, self.submat_in, self.submat_out)  # [E, submat_in, submat_out]
        node = x[edge_index[0]].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        out = self.propagate(edge_index=edge_index, x=msg, edge_weight=edge_norm)
        return out

    def message(self, x, edge_weight):
        if edge_weight is not None:
            return x * edge_weight
        return x

    def update(self, aggr_out):
        return aggr_out


class UnionRGCNLayer(MessagePassing):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__(aggr='add')

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, edge_index, edge_type, edge_norm, prev_h, emb_rel):
        self.rel_emb = emb_rel
        if self.self_loop:
            loop_message = torch.mm(x, self.evolve_loop_weight)
        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)

        node_repr = self.propagate(edge_index=edge_index,
                                   x=x,
                                   edge_type=edge_type,
                                   edge_weight=edge_norm)

        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        return node_repr

    def message(self, x_j, edge_type, edge_weight):
        relation = self.rel_emb.index_select(0, edge_type).view(-1, self.out_feat)
        msg = x_j + relation
        msg = torch.mm(msg, self.weight_neighbor)
        if edge_weight is not None:
            msg = msg * edge_weight
        return msg

    def update(self, aggr_out):
        return aggr_out


class UnionRGATLayer(MessagePassing):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGATLayer, self).__init__(aggr='add')
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # equation (2)
        self.attn_fc = nn.Linear(3 * self.out_feat, self.out_feat, bias=False)
        self.attn_fc2 = nn.Linear(self.out_feat, 1, bias=False)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, edge_index, edge_type, edge_norm, prev_h, emb_rel):
        self.rel_emb = emb_rel
        if self.self_loop:
            loop_message = torch.mm(x, self.evolve_loop_weight)
        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)

        # Compute attention scores
        row, col = edge_index
        src_h = x[row]
        dst_h = x[col]
        relation = self.rel_emb.index_select(0, edge_type).view(-1, self.out_feat)
        z2 = torch.cat([src_h, dst_h, relation], dim=1)
        e_att = F.leaky_relu(self.attn_fc2(self.attn_fc(z2)))

        # Normalize attention scores
        alpha = scatter(e_att, col, dim=0, dim_size=x.size(0), reduce='softmax')

        # Message passing
        msg = torch.cat([src_h, dst_h, relation], dim=1)
        msg = self.attn_fc(msg)
        msg = alpha * msg
        if edge_norm is not None:
            msg = msg * edge_norm.unsqueeze(-1)

        node_repr = scatter(msg, col, dim=0, dim_size=x.size(0), reduce='sum')

        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        return node_repr


class CompGCNLayer(MessagePassing):
    def __init__(self, in_feat, out_feat, num_rels, comp, num_bases=-1, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(CompGCNLayer, self).__init__(aggr='add')
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.comp = comp

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))
            nn.init.xavier_uniform_(self.skip_connect_weight, gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, edge_index, edge_type, edge_norm, prev_h, emb_rel):
        self.rel_emb = emb_rel
        if self.self_loop:
            loop_message = torch.mm(x, self.evolve_loop_weight)
        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)

        # Message passing
        node_repr = self.propagate(edge_index=edge_index, x=x, edge_type=edge_type, edge_weight=edge_norm)

        if prev_h is not None and len(prev_h) != 0 and self.skip_connect:
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        return node_repr

    def message(self, x_j, edge_type, edge_weight):
        relation = self.rel_emb.index_select(0, edge_type).view(-1, self.out_feat)
        node = x_j.view(-1, self.out_feat)
        if self.comp == "sub":
            msg = node + relation
        elif self.comp == "mult":
            msg = node * relation
        msg = torch.mm(msg, self.weight_neighbor)
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        return msg

    def update(self, aggr_out):
        return aggr_out


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, encoder_name="", opn="sub",
                 rel_emb=None, use_cuda=False, analysis=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_basis = num_basis
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.encoder_name = encoder_name
        self.use_cuda = use_cuda
        self.run_analysis = analysis
        self.rel_emb = rel_emb
        self.opn = opn
        # create rgcn layers
        self.build_model()
        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.features is not None:
            g.ndata['id'] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')


class MLPLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLPLinear, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x):
        x = self.act(F.normalize(self.linear1(x), p=2, dim=1))
        x = self.act(F.normalize(self.linear2(x), p=2, dim=1))

        return x


class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc,
                                  rel_emb=self.rel_emb)
        elif self.encoder_name == "kbat":
            return UnionRGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                                  activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc,
                                  rel_emb=self.rel_emb)
        elif self.encoder_name == "compgcn":
            return CompGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.opn, self.num_bases,
                                activation=act, self_loop=self.self_loop, dropout=self.dropout, skip_connect=sc,
                                rel_emb=self.rel_emb)
        else:
            raise NotImplementedError

    def forward(self, g, init_ent_emb, init_rel_emb):
        # g is a torch_geometric.data.Data
        if self.encoder_name in ["uvrgcn", "kbat", "compgcn"]:
            node_id = g.id.squeeze()
            x = init_ent_emb[node_id]  # initial node features
            r = init_rel_emb
            edge_index = g.edge_index
            edge_type = g.edge_type
            edge_norm = getattr(g, "edge_norm", None)
            for i, layer in enumerate(self.layers):
                if isinstance(layer, UnionRGCNLayer) or isinstance(layer, UnionRGATLayer) or isinstance(layer,
                                                                                                        CompGCNLayer):
                    x = layer(x, edge_index, edge_type, edge_norm, [], r[i])
                elif isinstance(layer, RGCNBlockLayer):
                    x = layer(x, edge_index, edge_type, edge_norm, [])
                else:
                    # fallback: assume same signature as UnionRGCNLayer
                    x = layer(x, edge_index, edge_type, edge_norm, [], r[i])
            return x
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                node_id = self.features.squeeze()
            else:
                node_id = g.id.squeeze()
            x = init_ent_emb[node_id]
            edge_index = g.edge_index
            edge_type = g.edge_type
            edge_norm = getattr(g, "edge_norm", None)
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(x, edge_index, edge_type, edge_norm, prev_h)
            else:
                for layer in self.layers:
                    x = layer(x, edge_index, edge_type, edge_norm, [])
                    prev_h = x
            return x


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, path="checkpoint.pth", verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        self.best_model_state = model.state_dict()
        torch.save(self.best_model_state, self.path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.path}")

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))


def sort_by_last_dim_with_neg1_last(arr):
    # arr: shape (B, N, 4)
    last_val = arr[:, :, 3]  # shape (B, N)

                               
    sort_key = torch.where(last_val == -1, torch.tensor(float('inf'), device=arr.device), last_val)

            
    sorted_indices = torch.argsort(sort_key, dim=1)

                                       
    B, N, _ = arr.shape
    batch_indices = torch.arange(B, device=arr.device).unsqueeze(1).expand(B, N)

              
    sorted_arr = arr[batch_indices, sorted_indices]

    return sorted_arr


def score_per_query(score, score_for_query):
    # pdb.set_trace()
    max_score_per_query = torch.zeros_like(score).scatter_reduce(0, score_for_query, score, reduce='amax',
                                                                 include_self=False)
    score_stable = score - max_score_per_query[score_for_query]
    score_stable = torch.exp(score_stable)
    sum_exp_per_query = torch.zeros_like(score).scatter_add(0, score_for_query, score_stable)
    softmax_score = score_stable / sum_exp_per_query[score_for_query]
    return softmax_score


def score_for_query(score, rows_of_false, r=1):
    K = score.shape[0]
    score = score.view(K)  # (K,)
    device = score.device
    unique_rows, inverse_index = torch.unique(rows_of_false, return_inverse=True)  # inverse_index: (K,)
    num_groups = unique_rows.shape[0]

                                                       
    max_per_group = torch.full((num_groups,), float('-inf'), device=device)
    max_per_group = max_per_group.scatter_reduce(0, inverse_index, score, reduce='amax', include_self=True)

                          
    score_shifted = score - max_per_group[inverse_index]  # (K,)
    score_exp = torch.exp(score_shifted)  # (K,)

                            
    sum_exp_per_group = torch.zeros(num_groups, device=device)
    sum_exp_per_group = sum_exp_per_group.scatter_add(0, inverse_index, score_exp)

                        
    score_softmax = score_exp / sum_exp_per_group[inverse_index]
    return score_softmax, inverse_index


class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0,
                 channels=3, kernel_size=3, use_bias=True):
        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                                     padding=int(math.floor(
                                         kernel_size / 2)))  
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, static_emb_ent, head, relation):
        batch_size = len(head)
        e1_embed = F.tanh(embedding).unsqueeze(1)
        el_embedding_all = F.tanh(static_emb_ent)
        rel_embedded = emb_rel[relation].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embed, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        cl_x = x
        x = torch.mm(x, el_embedding_all.transpose(1, 0))
        return x, cl_x


class TransductiveConvDecoder(torch.nn.Module):
    """Convolutional decoder for the lifecycle-memory transductive stream."""

    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0,
                 channels=50, kernel_size=3, use_bias=True):
        super(TransductiveConvDecoder, self).__init__()
        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                                     padding=int(math.floor(
                                         kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)

    def forward(self, embedding, emb_rel, triplets, his_emb, pre_weight, pre_type, partial_embeding=None):
        batch_size = len(triplets)
        if pre_type == "all":
            e1_embedded_all = F.tanh(embedding)
            e1_embedded = e1_embedded_all[triplets[:, 0]].unsqueeze(1)
            e1_embed = e1_embedded
        rel_embedded = emb_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embed, rel_embedded], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        cl_x = x
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, partial_embeding.transpose(1, 0))
        return x, cl_x


def masked_mean_pooling(chain_embedding, mask):
    B, N, D = mask.shape[0], mask.shape[1], chain_embedding.shape[2]  
    chain_embedding = chain_embedding.view(B, N, D)                                                   
    valid_mask = ~mask                               
    expanded_mask = valid_mask.unsqueeze(-1).expand(-1, -1, D)           
    valid_embeddings = chain_embedding * expanded_mask                   
    num_valid_tokens = valid_mask.sum(dim=1).clamp(min=1)  
    pooled = valid_embeddings.sum(dim=1) / num_valid_tokens.unsqueeze(-1)  


    all_invalid = ~valid_mask.any(dim=1)                                     

                                                             
    fallback = chain_embedding.view(B, N, D)[:, 0, :]  # shape: (128, 256)


    return pooled


class SoftPrototypeMemory(nn.Module):

    def __init__(self, num_prototypes, embedding_dim, temperature=0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.embedding_dim = embedding_dim
        self.temperature = temperature

                                 
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim))
        nn.init.xavier_normal_(self.prototypes)

    def forward(self, x):
                                  

                          
        x_norm = F.normalize(x, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)

                                            
        sim = torch.mm(x_norm, proto_norm.t()) / self.temperature

                                       
        attn_weights = F.softmax(sim, dim=-1)

                           
        retrieved_memory = torch.mm(attn_weights, self.prototypes)

                                        
                                       
        proto_sim = torch.mm(proto_norm, proto_norm.t())
        identity = torch.eye(self.num_prototypes, device=x.device)
        ortho_loss = torch.norm(proto_sim - identity, p='fro')

        return retrieved_memory, attn_weights, ortho_loss


class TemporalHistoryAttention(nn.Module):

    def __init__(self, entity_dim, relation_dim):
        super().__init__()
        self.entity_dim = entity_dim

               
        self.w_q = nn.Linear(relation_dim, entity_dim)                  
                                                             
        self.w_k = nn.Linear(entity_dim, entity_dim)               
        self.w_v = nn.Linear(entity_dim, entity_dim)                 

        self.attn_scale = 1.0 / (entity_dim ** 0.5)
        self.layer_norm = nn.LayerNorm(entity_dim)
        self.output_proj = nn.Linear(entity_dim, entity_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, chain_embedding, chain_mask, query_relation_emb):                                 
        Q = self.w_q(query_relation_emb).unsqueeze(1)                                         
        K = self.w_k(chain_embedding)
        V = self.w_v(chain_embedding)
                                                               
        scores = torch.bmm(Q, K.transpose(1, 2)) * self.attn_scale                                       
        if chain_mask is not None:
                                             
            scores = scores.masked_fill(chain_mask.unsqueeze(1), -1e9)

        attn_weights = F.softmax(scores, dim=-1)
                     
        context = torch.bmm(attn_weights, V).squeeze(1)  # (B, Dim)

        output = self.layer_norm(self.output_proj(context) + context)        
        return output


class AdaptiveGate(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, static_emb, proto_emb):
              
        concat = torch.cat([static_emb, proto_emb], dim=-1)
                     
        alpha = self.gate_net(concat)
            
        out = alpha * static_emb + (1 - alpha) * proto_emb
        return out


class DoradoModel(nn.Module):
    def __init__(self,
                 num_ent,
                 num_rel,
                 num_heads,
                 entity_dim,
                 relation_dim,
                 num_layers,
                 dropout=0.0,
                 word_embedding_path=None,
                 word_embedding=False,
                 residual=True,
                 device='cuda',
                 layer_norm=False,
                 chain_max_length=10,
                 time_length=14,
                 word_embedding_dim=768,
                 num_code=50,
                 dataset_name=None,
                 trans_hidden_dim=256,
                 trans_dropout=0.2,
                 trans_num_bases=128,
                 trans_num_basis=128,
                 trans_score_weight=0.9,
                 contrastive_temperature=0.03,
                 trans_input_dropout=0.2,
                 trans_hidden_dropout=0.2,
                 trans_feat_dropout=0.2,
                 use_static_background=True,
                 num_static_relations=0,
                 num_static_words=0
                 ):
        super(DoradoModel, self).__init__()
        self.activation = F.relu
        self.word_embedding = word_embedding
        self.residual = residual
        self.num_rel = num_rel
        self.num_ents = num_ent
        self.dataset_name = dataset_name
        self.trans_hidden_dim = trans_hidden_dim
        num_rel = num_rel * 2
        self.layer_norm = layer_norm
        self.weight_t = nn.Parameter(torch.randn(1, entity_dim))
        self.bias_t = nn.Parameter(torch.randn(1, entity_dim))
        self.device = device
        self.initializer_range = 0.02
        self.bn_entity = torch.nn.BatchNorm1d(entity_dim)
        self.bn_relation = torch.nn.BatchNorm1d(relation_dim)
        self.entity_dim = entity_dim
        self.chain_max_length = chain_max_length
        self.bn_1 = torch.nn.BatchNorm1d(entity_dim)
        self.bn_2 = torch.nn.BatchNorm1d(entity_dim)
        self.entity_down_proj = nn.Linear(word_embedding_dim, int(entity_dim / 4))
        self.relation_down_proj = nn.Linear(relation_dim, int(relation_dim / 4))
        self.time_projection = nn.Linear(1, int(entity_dim / 4))
        self.num_code = num_code
        self.time_length = time_length
        if self.word_embedding:
            if word_embedding_path and os.path.exists(word_embedding_path):
                self.entity_embedding = torch.tensor(np.load(word_embedding_path), dtype=torch.float).to(device)
                import logging
                logging.info(f"[DORADO Init] Loaded entity embeddings from {word_embedding_path}, shape={self.entity_embedding.shape}")
            else:
                import logging
                logging.warning(f"[DORADO Init] Entity embedding file not found at {word_embedding_path}, using random initialization")
                gamma = 6.0
                epsilon = 1.0
                embedding_range = (gamma + epsilon) / word_embedding_dim
                self.entity_embedding = torch.zeros(num_ent, word_embedding_dim, device=device)
                nn.init.uniform_(self.entity_embedding, a=-embedding_range, b=embedding_range)
        if self.word_embedding and (entity_dim != word_embedding_dim):
            self.project = nn.Linear(word_embedding_dim, entity_dim)
        else:
            self.project = nn.Linear(entity_dim, entity_dim)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_rel, relation_dim)).to(device)
        self.empty_embedding = nn.Parameter(torch.Tensor(1, entity_dim)).to(device)
        self.cls_embedding = nn.Parameter(torch.Tensor(4, entity_dim)).to(device)

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.empty_embedding, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.cls_embedding, gain=nn.init.calculate_gain('relu'))

        self.filling_embedding = nn.Parameter(torch.Tensor(1, entity_dim)).to(device)
        nn.init.xavier_uniform_(self.filling_embedding, gain=nn.init.calculate_gain('relu'))

        self.merge_layer = nn.Linear(entity_dim + relation_dim, entity_dim)                                                  
                                                          
        self.chain_input_proj = nn.Linear(entity_dim, entity_dim)
                      
        self.history_attn = TemporalHistoryAttention(entity_dim, relation_dim)
                  
        self.proto_memory = SoftPrototypeMemory(num_prototypes=self.num_code, embedding_dim=entity_dim)
                  
        self.adaptive_gate = AdaptiveGate(entity_dim)
                        
        self.scoring_layer1 = nn.Linear(self.entity_dim, self.entity_dim)
        self.scoring_layer2 = nn.Linear(self.entity_dim, 1)
        self.projection = nn.Linear(entity_dim * 2, entity_dim)
        self.relation_proj = nn.Linear(relation_dim, relation_dim)
        self.lstm_encoder = nn.LSTM(entity_dim, entity_dim, batch_first=True)
        self.mlp_encoder_1 = nn.Linear(entity_dim * 4, entity_dim * 2)
        self.mlp_encoder_2 = nn.Linear(entity_dim * 2, entity_dim)
        self.decoder = ConvTransE(num_ent, entity_dim, input_dropout=dropout, hidden_dropout=dropout,
                                  feature_map_dropout=dropout)

        # ========== DORADO lifecycle-memory transductive stream ==========
        self.use_transductive_stream = True
        self.use_static_background = use_static_background
        self.num_static_relations = num_static_relations
        self.num_static_words = num_static_words
        self.trans_num_bases = trans_num_bases
        self.trans_dropout = trans_dropout
        self.trans_state = None
        self.trans_recurrent_state = None
        if self.use_transductive_stream:
            self.trans_w1 = nn.Linear(self.trans_hidden_dim * 2, self.trans_hidden_dim)
            self.trans_w2 = nn.Linear(self.trans_hidden_dim, self.trans_hidden_dim)
            self.trans_w3 = nn.Linear(self.trans_hidden_dim, self.trans_hidden_dim)
            self.trans_w4 = nn.Linear(self.trans_hidden_dim * 2, self.trans_hidden_dim)
            self.trans_w6 = nn.Linear(self.trans_hidden_dim, self.trans_hidden_dim)
            self.trans_w7 = nn.Linear(self.trans_hidden_dim, self.trans_hidden_dim)
            self.trans_w_cl = nn.Linear(self.trans_hidden_dim * 2, self.trans_hidden_dim)

            self.trans_time_weight = nn.Parameter(torch.randn(1, self.trans_hidden_dim))
            self.trans_time_bias = nn.Parameter(torch.randn(1, self.trans_hidden_dim))

            pretrained_rel_emb = None
            pretrained_ent_emb = None
            pretrained_word_emb = None

            dataset_name = getattr(self, 'dataset_name', None)

            if dataset_name:
                if '__file__' in globals():
                    base_dir = os.path.dirname(__file__)
                else:
                    base_dir = os.getcwd()

                project_root = os.path.dirname(base_dir) if os.path.basename(base_dir) == 'code' else base_dir

                candidate_bases = [
                    os.path.join(os.getcwd(), 'data'),
                    os.path.join(base_dir, 'data'),
                    os.path.join(base_dir, '..', 'data'),
                    os.path.join(project_root, 'data'),
                ]

                rel_bert_path = None
                ent_bert_path = None
                word_bert_path = None

                # Find the first existing base directory
                for base in candidate_bases:
                    test_path = os.path.join(base, dataset_name, f'{dataset_name}_Bert_Entity_Embedding.npy')
                    if os.path.exists(test_path):
                        rel_bert_path = os.path.join(base, dataset_name, f'{dataset_name}_Bert_Relation_Embedding.npy')
                        ent_bert_path = os.path.join(base, dataset_name, f'{dataset_name}_Bert_Entity_Embedding.npy')
                        word_bert_path = os.path.join(base, dataset_name, f'{dataset_name}_Bert_Word_Embedding.npy')
                        break

                if rel_bert_path and os.path.exists(rel_bert_path):
                    pretrained_rel_emb = np.load(rel_bert_path)
                    import logging
                    logging.info(
                        f"[Context Init] Loaded BERT relation embeddings from {rel_bert_path}, shape={pretrained_rel_emb.shape}")
                    print(
                        f"[Context Init] Loaded BERT relation embeddings from {rel_bert_path}, shape={pretrained_rel_emb.shape}")
                if ent_bert_path and os.path.exists(ent_bert_path):
                    pretrained_ent_emb = np.load(ent_bert_path)
                    import logging
                    logging.info(
                        f"[Context Init] Loaded BERT entity embeddings from {ent_bert_path}, shape={pretrained_ent_emb.shape}")
                    print(
                        f"[Context Init] Loaded BERT entity embeddings from {ent_bert_path}, shape={pretrained_ent_emb.shape}")
                if word_bert_path and os.path.exists(word_bert_path):
                    pretrained_word_emb = np.load(word_bert_path)
                    import logging
                    logging.info(
                        f"[Context Init] Loaded BERT word embeddings from {word_bert_path}, shape={pretrained_word_emb.shape}")
                    print(
                        f"[Context Init] Loaded BERT word embeddings from {word_bert_path}, shape={pretrained_word_emb.shape}")

            self.trans_relation_bank = nn.Parameter(torch.Tensor(self.num_rel * 2, self.trans_hidden_dim),
                                              requires_grad=True).float().to(device)
            nn.init.xavier_normal_(self.trans_relation_bank)

            if pretrained_rel_emb is not None:
                rel_emb = torch.from_numpy(pretrained_rel_emb).float()  # [num_rels, d_bert]
                if rel_emb.shape[0] != self.num_rel:
                    raise ValueError(
                        f"Pretrained relation embedding num_rels mismatch: {rel_emb.shape[0]} vs {self.num_rel}")
                rel_cat = torch.cat([rel_emb, rel_emb], dim=0)  # [2*num_rels, d_bert]
                d_in = rel_cat.shape[1]
                if d_in == self.trans_hidden_dim:
                    with torch.no_grad():
                        self.trans_relation_bank.copy_(rel_cat)
                    import logging
                    logging.info(f"[Context Init] Initialized relation embeddings with BERT (direct copy)")
                    print(f"[Context Init] Initialized relation embeddings with BERT (direct copy)")
                else:
                    with torch.no_grad():
                        k = min(self.trans_hidden_dim, d_in)
                        U, S, Vt = torch.linalg.svd(rel_cat, full_matrices=False)
                        W = Vt[:k].T  # [d_in, k]
                        proj = rel_cat @ W  # [2*num_rels, k]
                        if k < self.trans_hidden_dim:
                            pad = torch.zeros(proj.shape[0], self.trans_hidden_dim - k, device=proj.device)
                            proj = torch.cat([proj, pad], dim=1)
                        self.trans_relation_bank.copy_(proj[:, :self.trans_hidden_dim])
                    import logging
                    logging.info(
                        f"[Context Init] Initialized relation embeddings with BERT (projected from {d_in} to {self.trans_hidden_dim})")
                    print(
                        f"[Context Init] Initialized relation embeddings with BERT (projected from {d_in} to {self.trans_hidden_dim})")

            self.trans_entity_bank = nn.Parameter(torch.Tensor(num_ent, self.trans_hidden_dim),
                                                  requires_grad=True).float().to(device)
            nn.init.normal_(self.trans_entity_bank)

            if pretrained_ent_emb is not None:
                emb = torch.from_numpy(pretrained_ent_emb).float()
                if emb.shape[0] != num_ent:
                    print(
                        f"[Context Init] Warning: BERT entity embedding num_ents mismatch: {emb.shape[0]} vs {num_ent}, using random init")
                else:
                    if emb.shape[1] == self.trans_hidden_dim:
                        with torch.no_grad():
                            self.trans_entity_bank.copy_(emb)
                        import logging
                        logging.info(f"[Context Init] Initialized entity embeddings with BERT (direct copy)")
                        print(f"[Context Init] Initialized entity embeddings with BERT (direct copy)")
                    else:
                        with torch.no_grad():
                            k = min(self.trans_hidden_dim, emb.shape[1])
                            U, S, Vt = torch.linalg.svd(emb, full_matrices=False)
                            W = Vt[:k].T  # [d, k]
                            proj = emb @ W  # [N, k]
                            if k < self.trans_hidden_dim:
                                pad = torch.zeros(proj.shape[0], self.trans_hidden_dim - k, device=proj.device)
                                proj = torch.cat([proj, pad], dim=1)
                            self.trans_entity_bank.copy_(proj[:, :self.trans_hidden_dim])
                        import logging
                        logging.info(
                            f"[Context Init] Initialized entity embeddings with BERT (projected from {emb.shape[1]} to {self.trans_hidden_dim})")
                        print(
                            f"[Context Init] Initialized entity embeddings with BERT (projected from {emb.shape[1]} to {self.trans_hidden_dim})")

            self.trans_word_bank = nn.Parameter(torch.Tensor(self.num_static_words, self.trans_hidden_dim),
                                                requires_grad=True).float().to(device)
            nn.init.xavier_normal_(self.trans_word_bank)
            if pretrained_word_emb is not None:
                w_emb = torch.from_numpy(pretrained_word_emb).float()  # [num_words, d_bert]
                if w_emb.shape[0] != self.num_static_words:
                    raise ValueError(
                        f"Pretrained word embedding num_words mismatch: {w_emb.shape[0]} vs {self.num_static_words}")
                d_in = w_emb.shape[1]
                if d_in == self.trans_hidden_dim:
                    with torch.no_grad():
                        self.trans_word_bank.copy_(w_emb)
                else:
                    with torch.no_grad():
                        k = min(self.trans_hidden_dim, d_in)
                        U, S, Vt = torch.linalg.svd(w_emb, full_matrices=False)
                        W = Vt[:k].T  # [d_in, k]
                        proj = w_emb @ W  # [num_words, k]
                        if k < self.trans_hidden_dim:
                            pad = torch.zeros(proj.shape[0], self.trans_hidden_dim - k, device=proj.device)
                            proj = torch.cat([proj, pad], dim=1)
                        self.trans_word_bank.copy_(proj[:, :self.trans_hidden_dim])
            self.static_rgcn_layer = RGCNBlockLayer(
                self.trans_hidden_dim, self.trans_hidden_dim, self.num_static_relations * 2, self.trans_num_bases,
                activation=F.rrelu, dropout=self.trans_dropout, self_loop=False, skip_connect=False
            ).to(device)

            self.trans_rgcn = RGCNCell(
                num_ent, self.trans_hidden_dim, self.trans_hidden_dim, num_rel,
                num_bases=trans_num_bases, num_basis=trans_num_basis, num_hidden_layers=num_layers,
                dropout=trans_dropout, self_loop=True, skip_connect=False,
                encoder_name="uvrgcn", opn="sub", rel_emb=self.trans_relation_bank,
                use_cuda=(device != 'cpu'), analysis=False
            )

            # transductive stream GRU for entity evolution
            self.trans_entity_cell = nn.GRUCell(self.trans_hidden_dim, self.trans_hidden_dim)

            # transductive stream relation time gate
            self.trans_time_gate_weight = nn.Parameter(torch.Tensor(self.trans_hidden_dim, self.trans_hidden_dim))
            nn.init.xavier_uniform_(self.trans_time_gate_weight, gain=nn.init.calculate_gain('relu'))
            self.trans_time_gate_bias = nn.Parameter(torch.Tensor(self.trans_hidden_dim))
            nn.init.zeros_(self.trans_time_gate_bias)

            self.trans_w_cl = nn.Linear(self.trans_hidden_dim * 2, self.trans_hidden_dim)
            self.contrastive_temperature = contrastive_temperature
            # MLPLinear: two linear layers with LeakyReLU (exactly like transductive stream)
            self.trans_projection_linear1 = nn.Linear(self.trans_hidden_dim, self.trans_hidden_dim)
            self.trans_projection_linear2 = nn.Linear(self.trans_hidden_dim, self.trans_hidden_dim)
            self.trans_projection_act = nn.LeakyReLU(0.2)
            self.use_temporal_contrastive = True
            self.history_fusion_mode = "all"
            self.trans_score_weight = trans_score_weight

            self.trans_decoder = TransductiveConvDecoder(num_ent, self.trans_hidden_dim, trans_input_dropout,
                                                         trans_hidden_dropout, trans_feat_dropout).to(device)  
                           
            self.trans_rel_attn = nn.Linear(self.trans_hidden_dim, 1).to(device)
                                 
            self.trans_semantic_proj = nn.Linear(entity_dim, self.trans_hidden_dim).to(device)
            self.trans_semantic_gate = nn.Linear(self.trans_hidden_dim * 2, self.trans_hidden_dim).to(device)        

            self.score_gate_dim = min(entity_dim, self.trans_hidden_dim)
            self.score_gate_trans_proj = nn.Linear(self.trans_hidden_dim, self.score_gate_dim)
            self.score_gate_ind_proj = nn.Linear(entity_dim, self.score_gate_dim)
            self.score_gate = nn.Sequential(
                nn.Linear(self.score_gate_dim * 2, self.score_gate_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.score_gate_dim, 1)
            )
            nn.init.zeros_(self.score_gate[-1].bias)
            self.last_score_gate_beta = None
            self.last_distill_stats = None
        print('init model')
        print('entity_embedding', self.entity_embedding.shape)
        print('relation_embedding', self.relation_embedding.shape)

    def e2r(self, triplets, num_rels):
                          
        # Convert to numpy if needed (original transductive stream expects numpy array)
        if isinstance(triplets, torch.Tensor):
            triplets = triplets.cpu().numpy()
        triplets = np.asarray(triplets)

        if triplets.shape[-1] > 3:
            triplets = triplets[:, :3]

        src, rel, dst = triplets.transpose()
        # get all relations
        # uniq_e = np.concatenate((src, dst))
        uniq_e = np.unique(src)
        # generate r2e
        e_to_r = defaultdict(set)
        for j, (src, rel, dst) in enumerate(triplets):
            e_to_r[src].add(rel)

        r_len = []
        r_idx = []
        idx = 0
        for e in uniq_e:
            r_len.append((idx, idx + len(e_to_r[e])))
            r_idx.extend(list(e_to_r[e]))
            idx += len(e_to_r[e])
        uniq_e = torch.from_numpy(np.array(uniq_e)).long().to(self.device)
        r_len = torch.from_numpy(np.array(r_len)).long().to(self.device)
        r_idx = torch.from_numpy(np.array(r_idx)).long().to(self.device)
        return uniq_e, r_len, r_idx

    def encode_transductive_stream(self, triples, history_glist, use_cuda, T_idx=None, static_graph=None):
        uniq_e, r_len, r_idx = self.e2r(triples, self.num_rel)

        temp_r = self.trans_relation_bank[r_idx]  # (Total_Rel_Edges, Dim)
                 
        # score: (Total_Rel_Edges, 1)
        rel_scores = self.trans_rel_attn(temp_r)

        e_input = torch.zeros(self.num_ents, self.trans_hidden_dim, device=self.device, dtype=torch.float)
                                         
        for span, e_idx in zip(r_len, uniq_e):
                           
            rel_vecs = temp_r[span[0]:span[1], :]
            scores = rel_scores[span[0]:span[1], :]

            attn_weights = F.softmax(scores, dim=0)
                             
            x_weighted = torch.sum(rel_vecs * attn_weights, dim=0, keepdim=True)
            e_input[e_idx] = x_weighted
                                                                             
        query_mask = torch.zeros((self.num_ents, self.trans_hidden_dim), device=self.device)
        t1 = torch.tensor(T_idx, device=self.device, dtype=torch.float)
        q_t = torch.cos(self.trans_time_weight * t1 + self.trans_time_bias).repeat(self.num_ents, 1)
        qe_emb = self.trans_w4(torch.cat([self.trans_entity_bank, q_t], dim=1))
        e1_emb = qe_emb[uniq_e]
        rel_emb = e_input[uniq_e]
        query_emb = self.trans_w1(torch.cat([e1_emb, rel_emb], dim=1))
        query_mask[uniq_e] = query_emb
                                
        if self.use_static_background and static_graph is not None:
            if static_graph.edge_index.device != self.device:
                static_graph = static_graph.to(self.device)
            static_x = torch.cat((self.trans_entity_bank, self.trans_word_bank), dim=0)
            static_emb_all = self.static_rgcn_layer(
                static_x,
                static_graph.edge_index,
                static_graph.edge_type,
                getattr(static_graph, "edge_norm", None),
                []
            )
            static_emb = static_emb_all[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb

            self.trans_state = static_emb
        else:
            self.trans_state = F.normalize(self.trans_entity_bank) if self.layer_norm else self.trans_entity_bank[:, :]
            static_emb = None
                                                                                        
        static_semantics = self.trans_semantic_proj(self.project(self.entity_embedding))  # (Num_Ent, trans_hidden_dim)

        his_r_emb = F.normalize(self.trans_relation_bank)
        history_embs = []
        att_embs = []
        his_temp_embs = []
        his_rel_embs = []

        if self.history_fusion_mode == "all":
            g_list = [g for g in history_glist if g is not None]

            for i, g in enumerate(g_list):
                if g.edge_index.device != self.device:
                    g = g.to(self.device)

                t2 = len(g_list) - i + 1
                t2_tensor = torch.tensor(t2, device=self.device, dtype=torch.float)
                h_t = torch.cos(self.trans_time_weight * t2_tensor + self.trans_time_bias).repeat(self.num_ents, 1)

                self.trans_state = self.trans_w4(torch.concat([self.trans_state, h_t], dim=1))

                temp_e = self.trans_state[g.r_to_e]

                                                                  
                x_input = torch.zeros(self.num_rel * 2, self.trans_hidden_dim, device=self.device, dtype=torch.float)
                                                      
                                                                                                              
                for span, r_idx in zip(g.r_len, g.uniq_r):
                    x = temp_e[span[0]:span[1], :]
                    x_mean = torch.mean(x, dim=0, keepdim=True)
                    x_input[r_idx] = x_mean

                x_input = self.trans_relation_bank + x_input
                num_layers = len(self.trans_rgcn.layers)
                rel_emb_list = [self.trans_relation_bank] * num_layers

                # RGCN Forward
                current_h = self.trans_rgcn.forward(g, self.trans_state, rel_emb_list)
                current_h = F.normalize(current_h) if self.layer_norm else current_h

                att_e = F.softmax(self.trans_w2(query_mask + current_h), dim=1)                                               
                        
                gate = torch.sigmoid(self.trans_semantic_gate(torch.cat([current_h, static_semantics], dim=1)))
                enhanced_input = gate * current_h + (1 - gate) * static_semantics

                if i == 0:
                    self.trans_recurrent_state = self.trans_entity_cell(enhanced_input, self.trans_state)
                else:
                    self.trans_recurrent_state = self.trans_entity_cell(enhanced_input, self.trans_recurrent_state)

                self.trans_recurrent_state = F.normalize(self.trans_recurrent_state) if self.layer_norm else self.trans_recurrent_state
                # --------------------------------

                time_weight = F.sigmoid(torch.mm(x_input, self.trans_time_gate_weight) + self.trans_time_gate_bias)
                hr = time_weight * x_input + (1 - time_weight) * self.trans_relation_bank
                hr = F.normalize(hr) if self.layer_norm else hr

                history_embs.append(self.trans_recurrent_state)
                his_rel_embs.append(hr)
                his_temp_embs.append(self.trans_recurrent_state)

                self.trans_state = self.trans_recurrent_state
                att_emb = att_e * self.trans_recurrent_state
                att_embs.append(att_emb.unsqueeze(0))

            att_ent = torch.mean(torch.concat(att_embs, dim=0), dim=0)
            att_ent = F.normalize(att_ent)
            history_emb = att_ent + history_embs[-1]
            history_emb = F.normalize(history_emb) if self.layer_norm else history_emb
        else:
            hr = None
            history_emb = None

        return history_emb, static_emb, hr, None, his_r_emb, his_temp_embs, his_rel_embs

    def projection_head(self, x):
        x = self.trans_projection_act(F.normalize(self.trans_projection_linear1(x), p=2, dim=1))
        x = self.trans_projection_act(F.normalize(self.trans_projection_linear2(x), p=2, dim=1))
        return x

    def compute_temporal_contrastive_loss(self, ent1_emb, ent2_emb):
        """Temporal contrastive loss for the lifecycle-memory transductive stream."""
        loss_fn = nn.CrossEntropyLoss().to(self.device)
        z1 = self.projection_head(ent1_emb)
        z2 = self.projection_head(ent2_emb)
        pred1 = torch.mm(z1, z2.T)
        pred2 = torch.mm(z2, z1.T)
        pred3 = torch.mm(z1, z1.T)
        pred4 = torch.mm(z2, z2.T)
        labels = torch.arange(pred1.shape[0]).to(self.device)

        train_cl_loss = (loss_fn(pred1 / self.contrastive_temperature, labels) +
                         loss_fn(pred2 / self.contrastive_temperature, labels) +
                         loss_fn(pred3 / self.contrastive_temperature, labels) +
                         loss_fn(pred4 / self.contrastive_temperature, labels)) / 4
        return train_cl_loss

    def fuse_embeddings(self, dynamic_emb, graph_emb):
        """Fuse inductive and transductive entity representations."""
        if not self.use_transductive_stream:
            return dynamic_emb

        if graph_emb is None:
            return dynamic_emb

        # Check shapes
        if dynamic_emb.shape != graph_emb.shape:
            # If shapes don't match, resize graph_emb to match dynamic_emb
            if graph_emb.shape[0] != dynamic_emb.shape[0]:
                # If number of entities doesn't match, use zeros
                graph_emb = torch.zeros_like(dynamic_emb)
            else:
                # If only feature dimension differs, pad or truncate
                if graph_emb.shape[1] < dynamic_emb.shape[1]:
                    padding = torch.zeros(graph_emb.shape[0], dynamic_emb.shape[1] - graph_emb.shape[1],
                                          device=graph_emb.device, dtype=graph_emb.dtype)
                    graph_emb = torch.cat([graph_emb, padding], dim=1)
                elif graph_emb.shape[1] > dynamic_emb.shape[1]:
                    graph_emb = graph_emb[:, :dynamic_emb.shape[1]]

        # Ensure same device
        if graph_emb.device != dynamic_emb.device:
            graph_emb = graph_emb.to(dynamic_emb.device)

        try:
            if self.fusion_method == 'concat':
                fused = torch.cat([dynamic_emb, graph_emb], dim=-1)
                fused = self.fusion_layer(fused)
                return F.normalize(fused) if self.layer_norm else fused
            elif self.fusion_method == 'attention':
                fused, _ = self.fusion_attention(
                    dynamic_emb.unsqueeze(1),
                    graph_emb.unsqueeze(1),
                    graph_emb.unsqueeze(1)
                )
                return fused.squeeze(1)
            elif self.fusion_method == 'gate':
                concat_emb = torch.cat([dynamic_emb, graph_emb], dim=-1)
                gate = self.fusion_gate(concat_emb)
                fused = gate * dynamic_emb + (1 - gate) * graph_emb
                return fused
            else:
                # Default: simple average
                return (dynamic_emb + graph_emb) / 2
        except Exception as e:
            # If fusion fails, return dynamic_emb as fallback
            print(f"Warning: Fusion failed, using inductive stream embedding only. Error: {e}")
            return dynamic_emb

    def forward(self, triples, chain_embedding, chain_mask, embedding_dict, history_glist=None, static_graph=None,
                use_cuda=True, T_idx=None):
        # torch.autograd.set_detect_anomaly(True)
        ground_truth = triples
        query_relation_embedding = embedding_dict['relation_embedding'][ground_truth[:, 1]]
        query_entity = ground_truth[:, 0]
        edge_type = ground_truth[:, 1]
        label = ground_truth[:, 2]
                                                   
        if chain_embedding.dim() == 4:
            chain_embedding = chain_embedding.mean(dim=2)  # (B, Seq, Hidden)

                        
        chain_feat = self.chain_input_proj(chain_embedding)  # (B, Seq, Dim)

                                          
        local_history_context = self.history_attn(chain_feat, chain_mask, query_relation_embedding)

                      
        query_ctx_embedding = self.bn_1(local_history_context)
        relation_vec = self.bn_relation(query_relation_embedding)
        final_query_embedding = query_ctx_embedding + relation_vec
                        
        static_entity_embedding = self.project(self.entity_embedding)

                                  
        context_aware_static_emb = static_entity_embedding.clone()
        context_aware_static_emb[query_entity] = context_aware_static_emb[query_entity] + final_query_embedding

               
        retrieved_proto, attn_weights, ortho_loss = self.proto_memory(context_aware_static_emb)

                 
        dynamic_entity_embedding = self.adaptive_gate(context_aware_static_emb, retrieved_proto)

        dynamic_entity_embedding = F.normalize(
            dynamic_entity_embedding) if self.layer_norm else dynamic_entity_embedding

        transductive_scores = None
        transductive_embedding = None
        transductive_relation_embedding = None
        transductive_relation_bank = None
        transductive_states = []
        transductive_relation_states = []
        trans_feat = None

                                
        if self.use_transductive_stream and history_glist is not None:
            transductive_embedding, static_emb, transductive_relation_embedding, his_emb, transductive_relation_bank, transductive_states, transductive_relation_states = self.encode_transductive_stream(
                triples, history_glist, use_cuda, T_idx=T_idx, static_graph=static_graph)
            if transductive_embedding is not None and transductive_relation_embedding is not None:
                transductive_scores, trans_feat = self.trans_decoder.forward(
                    transductive_embedding, transductive_relation_embedding, triples, None,
                    self.trans_score_weight, self.history_fusion_mode)

                                       
        inductive_scores, ind_feat = self.decoder.forward(final_query_embedding, self.relation_embedding,
                                                          dynamic_entity_embedding, query_entity, edge_type)

              
        if transductive_scores is not None:
            z_trans = self.score_gate_trans_proj(trans_feat)
            z_ind = self.score_gate_ind_proj(ind_feat)
            gate_input = torch.cat([z_trans, z_ind], dim=-1)
            beta_q = torch.sigmoid(self.score_gate(gate_input))  # [B, 1], broadcast over candidate entities.
            scores_ob = beta_q * transductive_scores + (1.0 - beta_q) * inductive_scores
            self.last_score_gate_beta = beta_q.detach()
        else:
            scores_ob = inductive_scores
            self.last_score_gate_beta = None
                                   
        temporal_contrastive_loss = torch.zeros(1).to(self.device)
        if self.use_transductive_stream and self.use_temporal_contrastive and self.history_fusion_mode == "all":
            if transductive_embedding is not None and transductive_states is not None and len(transductive_states) > 0 and transductive_relation_bank is not None:
                for idx, evolve_emb in enumerate(transductive_states):
                    query = torch.cat([transductive_embedding[triples[:, 0]], transductive_relation_bank[triples[:, 1]]], dim=1)
                    query2 = torch.cat([evolve_emb[triples[:, 0]], transductive_relation_states[idx][triples[:, 1]]], dim=1)
                    x1 = self.trans_w_cl(query)
                    x2 = self.trans_w_cl(query2)
                    temporal_contrastive_loss += self.compute_temporal_contrastive_loss(x1, x2)

        scores_en = F.log_softmax(scores_ob, dim=1)
        task_loss = F.nll_loss(scores_en, label)
                                                                                

        mutual_distill_loss = torch.zeros(1).to(self.device)
        self.last_distill_stats = {
            'batch_size': int(label.shape[0]),
            'trans_correct': 0.0,
            'ind_correct': 0.0,
            'both_correct': 0.0,
            'trans_guides_ind': 0.0,
            'ind_guides_trans': 0.0,
            'active_distill_count': 0.0,
            'active_kl_sum': 0.0,
            'weighted_kl_sum': 0.0,
            'distill_weight_sum': 0.0,
            'beta_sum': 0.0,
            'beta_sq_sum': 0.0,
            'beta_min': None,
            'beta_max': None,
        }

        if self.use_transductive_stream and transductive_scores is not None:
                                             

            logits_trans = transductive_scores

            logits_ind = inductive_scores

            T = 2.0

            trans_prob = F.softmax(logits_trans.detach(), dim=1)
            ind_prob = F.softmax(logits_ind.detach(), dim=1)
            trans_conf, pred_trans = torch.max(trans_prob, dim=1)
            ind_conf, pred_ind = torch.max(ind_prob, dim=1)


            is_trans_correct = (pred_trans == label)

            is_ind_correct = (pred_ind == label)

            trans_guides_ind = is_trans_correct & ((~is_ind_correct) | (trans_conf >= ind_conf))

            ind_guides_trans = is_ind_correct & ((~is_trans_correct) | (ind_conf > trans_conf))

            trans_weight = torch.where(
                is_ind_correct,
                (trans_conf - ind_conf).clamp(min=0.0),
                trans_conf
            ) * trans_guides_ind.float()

            ind_weight = torch.where(
                is_trans_correct,
                (ind_conf - trans_conf).clamp(min=0.0),
                ind_conf
            ) * ind_guides_trans.float()

            loss_t2i = F.kl_div(
                F.log_softmax(logits_ind / T, dim=1),
                F.softmax(logits_trans.detach() / T, dim=1),
                reduction='none'
            ).sum(dim=1)

            loss_i2t = F.kl_div(
                F.log_softmax(logits_trans / T, dim=1),
                F.softmax(logits_ind.detach() / T, dim=1),
                reduction='none'
            ).sum(dim=1)

            batch_size = max(int(label.shape[0]), 1)
            active_t2i = trans_guides_ind.float()
            active_i2t = ind_guides_trans.float()
            active_kl_sum = ((loss_t2i * active_t2i).sum() + (loss_i2t * active_i2t).sum()) * (T**2)
            weighted_kl_sum = 0.5 * (
                (loss_t2i * trans_weight).sum() +
                (loss_i2t * ind_weight).sum()
            ) * (T**2)
            mutual_distill_loss = weighted_kl_sum / batch_size

            with torch.no_grad():
                beta_flat = self.last_score_gate_beta.view(-1) if self.last_score_gate_beta is not None else None
                self.last_distill_stats = {
                    'batch_size': int(label.shape[0]),
                    'trans_correct': float(is_trans_correct.float().sum().item()),
                    'ind_correct': float(is_ind_correct.float().sum().item()),
                    'both_correct': float((is_trans_correct & is_ind_correct).float().sum().item()),
                    'trans_guides_ind': float(active_t2i.sum().item()),
                    'ind_guides_trans': float(active_i2t.sum().item()),
                    'active_distill_count': float((active_t2i + active_i2t).sum().item()),
                    'active_kl_sum': float(active_kl_sum.item()),
                    'weighted_kl_sum': float(weighted_kl_sum.item()),
                    'distill_weight_sum': float((trans_weight.sum() + ind_weight.sum()).item()),
                    'beta_sum': float(beta_flat.sum().item()) if beta_flat is not None else 0.0,
                    'beta_sq_sum': float((beta_flat * beta_flat).sum().item()) if beta_flat is not None else 0.0,
                    'beta_min': float(beta_flat.min().item()) if beta_flat is not None and beta_flat.numel() > 0 else None,
                    'beta_max': float(beta_flat.max().item()) if beta_flat is not None and beta_flat.numel() > 0 else None,
                }

        # loss = task_loss + 0.1 * ortho_loss + temporal_contrastive_loss + 0.1 * mutual_distill_loss
        loss = task_loss + ortho_loss + temporal_contrastive_loss + 0.5*mutual_distill_loss
                                     
        return scores_en, loss, task_loss, ortho_loss, temporal_contrastive_loss, mutual_distill_loss

