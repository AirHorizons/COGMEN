import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .GNN import GNN
from .Classifier import Classifier
from .functions import batch_graphify
import cogmen

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class COGMEN(nn.Module):
    def __init__(self, args, log):
        super(COGMEN, self).__init__()
        u_dim = 100
        if args.rnn == "transformer":
            g_dim = args.hidden_size
        else:
            g_dim = 200
        h1_dim = args.hidden_size
        h2_dim = args.hidden_size
        hc_dim = args.hidden_size
        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
            "iemocap_4": {"hap": 0, "sad": 1, "neu": 2, "ang": 3},
            "mosei": {"Negative": 0, "Positive": 1},
        }

        dataset_speaker_dict = {
            "iemocap": 2,
            "iemocap_4": 2,
            "mosei": 1,
        }

        if args.dataset and args.emotion == "multilabel":
            dataset_label_dict["mosei"] = {
                "happiness": 0,
                "sadness": 1,
                "anger": 2,
                "surprise": 3,
                "disgust": 4,
                "fear": 5,
            }

        tag_size = len(dataset_label_dict[args.dataset])
        args.n_speakers = dataset_speaker_dict[args.dataset]
        self.n_speakers = args.n_speakers
        self.concat_gin_gout = args.concat_gin_gout

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.architecture = args.architecture
        self.global_node_type = args.global_node_type
        self.global_aggregate_method = args.global_aggregate_method
        if self.global_aggregate_method == 'weight':
            self.global_node_weight = nn.Parameter(torch.ones(g_dim, u_dim))
        elif self.global_aggregate_method == 'linear':
            self.global_node_linear = nn.Linear(g_dim, u_dim)

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.gcn = GNN(g_dim, h1_dim, h2_dim, args)
        if args.concat_gin_gout:
            self.clf = Classifier(
                g_dim + h2_dim * args.gnn_nheads, hc_dim, tag_size, args
            )
        else:
            self.clf = Classifier(h2_dim * args.gnn_nheads, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + "0"] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + "1"] = len(edge_type_to_idx)
        if 'single_global_node' in self.architecture:
            edge_type_to_idx["single_global_node"] = len(edge_type_to_idx)
        elif 'multiple_global_node' in self.architecture:
            edge_type_to_idx["global_node_self"] = len(edge_type_to_idx)
            # edge_type_to_idx["global_node_other"] = len(edge_type_to_idx)
        elif 'all_global_node' in self.architecture:
            edge_type_to_idx["single_global_node"] = len(edge_type_to_idx)
            edge_type_to_idx["global_node_self"] = len(edge_type_to_idx)
            # edge_type_to_idx["global_node_other"] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx

        self.log = log
        self.log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        # [batch_size, mx_len, D_g]
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])

        if 'single_global_node' in self.architecture:
            # Add global node features
            if self.global_aggregate_method == 'weight':
                global_node_features = torch.matmul(node_features, self.global_node_weight)
                global_node_features = torch.mean(global_node_features, dim=1)
                global_node_features = global_node_features.unsqueeze(1)
                node_features = torch.cat([node_features, global_node_features], dim=1)
            elif self.global_aggregate_method == 'linear':
                global_node_features = self.global_node_linear(node_features)
                global_node_features = torch.mean(global_node_features, dim=1)
                global_node_features = global_node_features.unsqueeze(1)
                node_features = torch.cat([node_features, global_node_features], dim=1)
            else: # self.global_aggregate_method == 'mean':
                # print(f'original node size: {node_features.shape}')
                
                # take average of node features in time dimension according to txt_len_tensor
                max_len = node_features.size(1)
                mask = torch.arange(max_len).to(device=self.device).expand(len(data["text_len_tensor"]), max_len) < data["text_len_tensor"].unsqueeze(1)
                mask = mask.unsqueeze(-1).expand(-1, -1, node_features.size(2))

                global_mask = node_features * mask.float()
                global_sum = global_mask.sum(dim=1)
                global_length = data["text_len_tensor"].float().unsqueeze(-1)
                global_node_features = global_sum / global_length

                # global_node_features = torch.mean(node_features, dim=1)  # Averaging across time dimension
                
                # global_node_features = torch.mean(node_features, dim=1)  # Averaging across time dimension
                
                global_node_features = global_node_features.unsqueeze(1)  # Add an extra dimension to match shape requirements.to(self.device)
                node_features = torch.cat([node_features, global_node_features], dim=1)
                # print(f'appended node size: {node_features.shape}\nglobal node size: {global_node_features.shape}')
                # print(f'length tensor: {data["text_len_tensor"]}')
        
        # print data_text_len_tensor
        # print(data["text_len_tensor"])
        # # print last 10 time steps of node_features
        # print(node_features[:3].shape)
        # print(f'node_features: {node_features[:3, -10:, :3]}')
        # quit()

        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
            self.architecture,
            self.n_speakers,
        )
        '''
        [[1657, 1659, 1650, 1667, 1661, 1663, 1654, 1655, 1672, 1663, 1664, 1666,
         1660, 1652, 1669, 1653, 1656, 1670, 1649, 1648, 1650, 1657, 1658, 1665,
         1659, 1667, 1661, 1663, 1654, 1664, 1655, 1672,   31,   31,   31,   31,
           31,   31,   31,   31,   31,   31,   31,   31,   31,   31,   31,   31,
           31,   31,   31,   31,   31,   31,   31,   31,   31,   31,   31,   31,
           31,   31,   31],...]
        '''
        # print(edge_index[:, 1128:1160])
        # print(edge_index.shape, edge_index_lengths.shape, edge_type.shape)
        # quit()

        graph_out = self.gcn(features, edge_index, edge_type)
        # remove global node features
        if 'single_global_node' in self.architecture:
            # print(graph_out.shape, features.shape)
            lengths = data["text_len_tensor"]
            global_node_indices = [sum(lengths[:i+1]) for i in range(len(lengths))]
            
            mask = torch.zeros(graph_out.shape[0], dtype=torch.bool)

            # debugging code
            # print(f'global node indices = {global_node_indices}')
            for i, indices in enumerate(global_node_indices):
                mask[indices] = True
            graph_out = graph_out[~mask]
            features = features[~mask]


        return graph_out, features

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        if self.concat_gin_gout:
            out = self.clf(
                torch.cat([features, graph_out], dim=-1), data["text_len_tensor"]
            )
        else:
            out = self.clf(graph_out, data["text_len_tensor"])

        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        # remove comment
        # print(graph_out.shape, features.shape)
        if self.concat_gin_gout:
            loss = self.clf.get_loss(
                torch.cat([features, graph_out], dim=-1),
                data["label_tensor"],
                data["text_len_tensor"],
            )
        else:
            loss = self.clf.get_loss(
                graph_out, data["label_tensor"], data["text_len_tensor"]
            )

        return loss
