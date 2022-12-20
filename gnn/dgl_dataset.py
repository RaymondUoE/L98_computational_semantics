import dgl
from dgl.data import DGLDataset
import torch
import pandas as pd
import json
from tqdm import tqdm
import delphin.codecs.eds
from featureriser import Featureriser
import sys
 
# setting path
sys.path.append('../')
from utils import string_of_list_to_list

class EdsDataset(DGLDataset):
    def __init__(self, name='random', save_dir='./data/processed', mode='train'):
        
        self.unknown_label = '<UNK>'
        self.mode = mode
        self.datapath = f'./data/raw/gnn_data_dgl_{self.mode}_small.csv'
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.save_path = './data/processed'
        super().__init__(name='eds')
    

    def process(self):
        data_to_be_processed = pd.read_csv(self.datapath).reset_index()

        num_of_data = len(data_to_be_processed)
        self.graphs = []

        print('Processing data...')
        if self.mode == 'train':
            self.label_dict = self._build_label_dict(list(data_to_be_processed['fn_frame'].values))
            list_of_edge_labels = list(data_to_be_processed.apply(lambda x: string_of_list_to_list(x['fn_roles']), axis=1))
            flattened_list = [y for x in list_of_edge_labels for y in x]
            self.edge_dict = self._build_edge_label_dict(flattened_list)
        else:
            with open('./data/node_label_dict.json', 'r') as f:
                self.label_dict = json.load(f)
                f.close()
            with open('./data/edge_label_dict.json', 'r') as f:
                self.edge_dict = json.load(f)
                f.close()
        
        edses = []
        sentences = []
        for index, row in data_to_be_processed.iterrows():
            edses.append(delphin.codecs.eds.decode(row['eds']))
            sentences.append(row['sentence'])
        all_graph_node_feature = Featureriser.bert_featurerise(edses, sentences)

        for index, row in tqdm(data_to_be_processed.iterrows(), total=data_to_be_processed.shape[0]):
            eds_str = row['eds']
            eds = delphin.codecs.eds.decode(eds_str)
            node_id_to_idx_dict = self._build_node_dict(eds.nodes)
            target_node = row['target_node']
                
            node_features_dict = all_graph_node_feature[index]
            nodes_embeds = []
            for n in eds.nodes:
                nodes_embeds.append(node_features_dict[n.id])
            node_features = torch.cat(nodes_embeds) # node features
            # print(node_features.shape)
            
            mask = [False if not target_node == x else True for x in node_id_to_idx_dict.keys()]
    
            node_labels = torch.tensor([-1 if not x else self._get_node_label_index(row['fn_frame']) for x in list(mask)]).to(self.device)
            
            
            edges_src, edges_tgt = self._eds_to_graph(eds, node_id_to_idx_dict)
            edge_features = self._get_edge_features(edges_src, edges_tgt, nodes_embeds).to(self.device) # edge features
            # print(edge_features.shape)

            graph = dgl.graph((edges_src, edges_tgt), num_nodes=len(eds.nodes)).to(self.device)
            graph.ndata['feat'] = node_features
            graph.ndata['label'] = node_labels
            graph.edata['weight'] = edge_features

            graph.ndata['mask'] = torch.tensor(mask).to(self.device)
            
            graph = dgl.add_reverse_edges(graph)
            graph = dgl.add_self_loop(graph)
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

        label_dict[self.unknown_label] = len(label_dict)

        print('Number of node labels: ', len(label_dict))
        with open('./data/node_label_dict.json', 'w') as f:
            f.write(json.dumps(label_dict, indent=2))
            f.close()
        return label_dict
    
    def _build_edge_label_dict(self, edge_labels):
        
        print('Building edge label dictionary...')
        label_dict = {}
        unique_labels = list(set(edge_labels))
        for l, ind in zip(unique_labels, range(len(unique_labels))):
            label_dict[l] = ind

        label_dict[self.unknown_label] = len(label_dict)
        
        print('Number of edge labels: ', len(label_dict))
        with open('./data/edge_label_dict.json', 'w') as f:
            f.write(json.dumps(label_dict, indent=2))
            f.close()
        return label_dict
    
    def _build_node_dict(self, nodes):
        nodes_to_idx_dict = {}

        for n, my_index in zip(nodes, range(len(nodes))):
            # id to index
            nodes_to_idx_dict[n.id] = my_index
        return nodes_to_idx_dict

    def _get_node_features(self, nodes):
        # out of date, not used, for reference only
        # return |V| x Dv
        torch.manual_seed(22)

        return torch.randn([len(nodes), 100],)
    
    def _get_edge_features(self, edges_src, edges_tgt, nodes_embeds):
        # return |E| x De
        # torch.manual_seed(22)
        # return torch.randn([len(edges), 50])
        edge_features = []
        for i in range(len(edges_src)):
            src_embed = nodes_embeds[edges_src[i]]
            tgt_embed = nodes_embeds[edges_tgt[i]]
            edge_features.append(torch.cat([src_embed, tgt_embed], dim=1).to(self.device))
        return torch.cat(edge_features, dim=0).to(self.device)

    def _eds_to_graph(self, eds, node_id_to_index_dict):
        
        edges_src = []
        edges_tgt = []
        for n in eds.nodes:
            for k, v in n.edges.items():
                edges_src.append(node_id_to_index_dict[n.id])
                edges_tgt.append(node_id_to_index_dict[v])
                
        return edges_src, edges_tgt

    def _get_node_label_index(self, label):
        if label in self.label_dict:
            return self.label_dict[label]
        else:
            return self.label_dict[self.unknown_label]

