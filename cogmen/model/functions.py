import numpy as np
import torch

import cogmen


def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, device, architecture, n_speakers=2):
    node_features, edge_index, edge_type = [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    for j in range(batch_size):
        global_offset = 1 if 'single_global_node' in architecture else 0
        cur_len = lengths[j].item()
        if 'single_global_node' in architecture:
            # append last node as global node
            # print(features[j, :cur_len, :].shape)
            # print(features[j, -1, :].unsqueeze(0).shape)
            # print(features[j, -1, :10])
            # print(features[j, -2, :10])
            feature = torch.cat([features[j, :cur_len, :], features[j, -1, :].unsqueeze(0)], dim = 0)
        else:
            feature = features[j, :cur_len, :]
        node_features.append(feature)
        perms = edge_perms(cur_len, wp, wf)
        # print(perms)
        # print(length_sum)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum) for item in perms]
        length_sum += cur_len + global_offset
        edge_index_lengths.append(len(perms) + global_offset * cur_len)
        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = "0"
            else:
                c = "1"
            edge_type.append(edge_type_to_idx[str(speaker1) + str(speaker2) + c])

        if 'single_global_node' in architecture:
            # adjust offset for global node
            global_node_idx = length_sum - 1
            for node_idx in range(cur_len):
                # Connection from global node to current node
                edge_index.append(torch.tensor([global_node_idx, global_node_idx - cur_len + node_idx]))
                edge_type.append(edge_type_to_idx["single_global_node"])
        # print(edge_index[:10])
        # print(edge_index[-10:])
        # print(edge_index)
        # print(edge_type)

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[: min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past) :]
        else:
            eff_array = array[
                max(0, j - window_past) : min(length, j + window_future + 1)
            ]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)
