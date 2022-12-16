import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import json
from tqdm import tqdm
import delphin.codecs.eds
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
# RAW_PATH = './data/raw/gnn_data_small.csv'

class EdsDataset(DGLDataset):
    def __init__(self, name='random', save_dir='./data/processed', mode='train'):
        

        self.mode = mode
        self.datapath = f'./data/raw/gnn_data_dgl_{self.mode}_small.csv'
        # self.save_path = './data/processed'
        super().__init__(name='eds')

    

    def process(self):
        data_to_be_processed = pd.read_csv(self.datapath).reset_index()

        num_of_data = len(data_to_be_processed)
        self.graphs = []

        print('Processing data...')
        if self.mode == 'train':
            self.label_dict = self._build_label_dict(list(data_to_be_processed['fn_frame'].values))
        else:
            with open('node_label_dict.json', 'r') as f:
                self.label_dict = json.load(f)
                f.close()

        for index, row in tqdm(data_to_be_processed.iterrows(), total=data_to_be_processed.shape[0]):
            eds_str = row['eds']
            eds = delphin.codecs.eds.decode(eds_str)
            node_id_to_idx_dict = self._build_node_dict(eds.nodes)
            target_node = row['target_node']
                
            node_features = self._get_node_features(eds.nodes)
            edge_features = self._get_edge_features(eds.nodes, eds.edges, node_id_to_idx_dict)
            mask = [False if not target_node == x else True for x in node_id_to_idx_dict.keys()]
    
            node_labels = torch.tensor([-1 if not x else self._get_node_label_index(row['fn_frame']) for x in list(mask)])
            
            
            edges_src, edges_tgt = self._eds_to_graph(eds, node_id_to_idx_dict)
            

            graph = dgl.graph((edges_src, edges_tgt), num_nodes=len(eds.nodes))
            graph.ndata['feat'] = node_features
            graph.ndata['label'] = node_labels
            graph.edata['weight'] = edge_features

            graph.ndata['mask'] = torch.tensor(mask)
            self.graphs.append(graph)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)

    def _build_label_dict(self, labels):
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
    
    def _build_node_dict(self, nodes):
        nodes_to_idx_dict = {}

        for n, my_index in zip(nodes, range(len(nodes))):
            # id to index
            nodes_to_idx_dict[n.id] = my_index
        return nodes_to_idx_dict

    def _get_node_label_index(self, label_dict, label):
        return label_dict[label]

    def _get_node_features(self, nodes):
        # TODO
        # return |V| x Dv
        return torch.randn([len(nodes), 100])
    
    def _get_edge_features(self, nodes, edges, node_id_to_idx_dict):
        # TODO
        # return |E| x De
        return torch.randn([len(edges), 50])

    def _eds_to_graph(self, eds, node_id_to_index_dict):
        
        edges_src = []
        edges_tgt = []
        for n in eds.nodes:
            for k, v in n.edges.items():
                edges_src.append(node_id_to_index_dict[n.id])
                edges_tgt.append(node_id_to_index_dict[v])
                
        return edges_src, edges_tgt

    def _get_node_label_index(self, label):
        if label in self.label_dict.keys():
            return self.label_dict[label]
        else:
            return self.label_dict['<UNK>']

    # def save(self):
    #     # save graphs and labels
    #     graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    #     save_graphs(graph_path, self.graphs)
    #     # # save other information in python dict
    #     # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    #     # save_info(info_path, {'num_classes': self.num_classes})

    # def load(self):
    #     # load processed data from directory `self.save_path`
    #     graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    #     self.graphs, label_dict = load_graphs(graph_path)
    #     # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    #     # self.num_classes = load_info(info_path)['num_classes']

    # def has_cache(self):
    #     # check whether there are processed data in `self.save_path`
    #     graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
    #     # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    #     return os.path.exists(graph_path) 

# dataset = EdsDataset(mode='train')
# graph = dataset[0]

# print(len(dataset.label_dict))