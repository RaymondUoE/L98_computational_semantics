import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np 
import os
import networkx as nx
import delphin.codecs.eds
import json

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")



class EdsDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename
        super(EdsDataset, self).__init__(root, transform, pre_transform)

# TODO
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

# TODO
    @property 
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

# TODO
    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        data_list = []

        self.label_dict = self._build_label_dict(list(self.data['fn_frame'].values))
        

        # edses = delphin.codecs.eds.loads('\n'.join(list(self.data['eds'].values)))
        # nxes = self._eds_to_networkx_batch(edses)
        for index, row in self.data.iterrows():
            eds_str = row['eds']

        # for eds_str in self.data['eds'].values:
            eds = delphin.codecs.eds.decode(eds_str)

            nodes, edges, mask = self._eds_to_geograph(eds, row['target_node'])
            x = torch.stack(nodes).squeeze()
            edge_index = torch.tensor(edges)
            data = Data(x=x, 
                        edge_index=edge_index.t().contiguous(),
                        mask = torch.tensor(mask),
                        y = torch.tensor([-1 if not x else self._get_node_label_index(self.label_dict, row['fn_frame']) for x in list(mask)]))
            data_list.append(data)




            # data = Data(x=node_feats, 
            #         edge_index=edge_index,
            #         edge_attr=edge_feats,
            #         y=label
            #         # smiles=mol["smiles"]
            #         ) 
        # featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        # for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
        #     # Featurize molecule
        #     mol = Chem.MolFromSmiles(row["smiles"])
        #     f = featurizer._featurize(mol)
        #     data = f.to_pyg_graph()
        #     data.y = self._get_label(row["HIV_active"])
        #     data.smiles = row["smiles"]
        #     if self.test:
        #         torch.save(data, 
        #             os.path.join(self.processed_dir, 
        #                          f'data_test_{index}.pt'))
        #     else:
        #         torch.save(data, 
        #             os.path.join(self.processed_dir, 
        #                          f'data_{index}.pt'))
            

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))        
        return data

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
    
    def _eds_to_geograph(self, eds, target_node):
        nodes_to_idx_dict = {}

        nodes = []
        for n, my_index in zip(eds.nodes, range(len(eds.nodes))):
            nodes_to_idx_dict[n.id] = my_index
            nodes.append(self._generate_feature(n))

        edges = []
        for n in eds.nodes:
            for k, v in n.edges.items():
                edge = [nodes_to_idx_dict[n.id], nodes_to_idx_dict[v]]
                edges.append(edge)

        return nodes, edges, [False if not target_node == x else True for x in nodes_to_idx_dict.keys()]

    def _generate_feature(self, node):
        # TODO
        return torch.randn([1,100])
    
    def _build_label_dict(self, labels):
        print('Building label dictionary...')
        label_dict = {}
        unique_labels = list(set(labels))
        for l, ind in zip(unique_labels, range(len(unique_labels))):
            label_dict[l] = ind

        print('Number of node labels: ', len(unique_labels))
        with open('./node_label_dict.json', 'w') as f:
            f.write(json.dumps(label_dict, indent=2))
            f.close()
        return label_dict

    def _get_node_label_index(self, label_dict, label):
        return label_dict[label]