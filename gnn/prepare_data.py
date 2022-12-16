import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data, InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np 
import os
import networkx as nx
import delphin.codecs.eds
import json

from sklearn.utils import shuffle
from tqdm import tqdm

PROCESSED_FILE_NAMES = ['processed_data_train.pt','processed_data_val.pt','processed_data_test.pt']
PROCESSED_FILE_PATH = './data/processed'
RAW_DATA_NAME = 'gnn_data_small.csv'
RAW_DATA_PATH = './data/raw'

def main():
    
    
    data = pd.read_csv(os.path.join(RAW_DATA_PATH, RAW_DATA_NAME)).reset_index()
    # data = shuffle(data, random_state=100)
    num_of_data = len(data)
    val_index = int(num_of_data * 0.8)
    test_index = int(num_of_data * 0.9)

    label_dict = _build_label_dict(list(data['fn_frame'].values))
    

    # edses = delphin.codecs.eds.loads('\n'.join(list(data['eds'].values)))
    # nxes = _eds_to_networkx_batch(edses)

    print('Processing data...')
    train_data = data[:val_index]
    val_data = data[val_index:test_index]
    test_data = data[test_index:]

    for i, data_to_be_processed in zip(range(3), [train_data, val_data, test_data]):
        data_list = []
        for index, row in tqdm(data_to_be_processed.iterrows(), total=data_to_be_processed.shape[0]):
            eds_str = row['eds']

        # for eds_str in data['eds'].values:
            eds = delphin.codecs.eds.decode(eds_str)

            nodes, edges, mask = _eds_to_geograph(eds, row['target_node'])
            x = torch.stack(nodes).squeeze()
            edge_index = torch.tensor(edges)
            data = Data(x=x, 
                        edge_index=edge_index.t().contiguous(),
                        mask = torch.tensor(mask),
                        y = torch.tensor([-1 if not x else _get_node_label_index(label_dict, row['fn_frame']) for x in list(mask)]))
            data_list.append(data)

        # data, slices = collate(data_list)
        # print(len(data))
        torch.save(data_list, os.path.join(PROCESSED_FILE_PATH, PROCESSED_FILE_NAMES[i]))
    # torch.save(data, os.path.join(processed_dir, f'processed_data_{mode}.pt'))
    # torch.save(data_list, os.path.join(processed_dir, f'processed_data_{mode}.pt'))



def _eds_to_networkx_batch(edses):
    nxes = []
    for eds in edses:

        G = nx.DiGraph()
        for node in eds.nodes:
            G.add_node(node.id, label = node.predicate)
            for e, t in node.edges.items():
                G.add_edge(e, t)
        
        nxes.append(G)
    return nxes

def _eds_to_geograph(eds, target_node):
    nodes_to_idx_dict = {}

    nodes = []
    for n, my_index in zip(eds.nodes, range(len(eds.nodes))):
        nodes_to_idx_dict[n.id] = my_index
        nodes.append(_generate_feature(n))

    edges = []
    for n in eds.nodes:
        for k, v in n.edges.items():
            edge = [nodes_to_idx_dict[n.id], nodes_to_idx_dict[v]]
            edges.append(edge)

    return nodes, edges, [False if not target_node == x else True for x in nodes_to_idx_dict.keys()]

def _generate_feature(node):
    # TODO
    return torch.randn([1,100])

def _build_label_dict(labels):
    print('Building label dictionary...')
    label_dict = {}
    unique_labels = list(set(labels))
    for l, ind in zip(unique_labels, range(len(unique_labels))):
        label_dict[l] = ind

    label_dict['<UNK>'] = len(label_dict)

    print('Number of node labels: ', len(unique_labels) + 1)
    with open('./node_label_dict.json', 'w') as f:
        f.write(json.dumps(label_dict, indent=2))
        f.close()
    return label_dict

def _get_node_label_index(label_dict, label):
    return label_dict[label]



if __name__ == "__main__":
    main()