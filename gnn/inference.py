from dgl_model import GCN, NodeLevelGNN
from dgl_dataset import EdsDataset
from dgl.dataloading import GraphDataLoader

import pickle
import torch
import copy
import json
import tqdm

import pandas as pd
import pytorch_lightning as pl

from utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():
    df = pd.read_csv('../sentences.csv')[:2000]
    df = df.dropna(how='any', axis=0)
    
    edses = []
    sentences = []
    ids = []
    errors = []
    print('Collecting and filtering data')
    for index, row in df.iterrows():
        try:
            edses.append(eds_from_string(row['eds']))
            sentences.append(row['sentence'])
            ids.append(row['id'])
        except:
            ids.append(row['id'])
    
    # with open('./data/prediction/pred_dataset.pkl', 'rb') as f:
    #     dataset = pickle.load(f)
    #     f.close
        
    dataset = EdsDataset(mode='pred', edses=edses, sentences=sentences)
    with open('./data/prediction/pred_dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
        f.close()
        
    predict_dataloader = GraphDataLoader(
        dataset, batch_size=1, drop_last=False)
    
    
    model = NodeLevelGNN.load_from_checkpoint("model/NodeLevelGCN/lightning_logs/version_66/checkpoints/epoch=71-step=37152.ckpt")
    model.to(device)
    model.eval()
    trainer = pl.Trainer(accelerator="gpu")
    
    predictions = trainer.predict(model, predict_dataloader)
    
    with open('./data/node_label_dict.json', 'r') as f:
        label_dict = json.load(f)
        f.close()
    with open('./data/edge_label_dict.json', 'r') as f:
        edge_dict = json.load(f)
        f.close()
    label_dict_decode = {v: k for k, v in label_dict.items()}
    edge_dict_decode = {v: k for k, v in edge_dict.items()}
    
    new_edses = []    
    pointer = 0
    out_dict = {}
    for i, eds in enumerate(edses):
        enhanced = copy.deepcopy(eds)
        
        target_nodes = [x.id for x in eds.nodes if '_v' in x.predicate]
        for target_node in target_nodes:
            # for eds such that 2 edges going to 1 target
            seen = []
            out_edges = [x.edges for x in eds.nodes if x.id == target_node][0]
            edge_targets = [b for a, b in out_edges.items() if 'ARG' in a]
            verb_pred_label = label_dict_decode[int(predictions[pointer]['verb_pred'][0])]
            
            enhanced_node = [x for x in enhanced.nodes if x.id == target_node][0]
            enhanced_node.predicate = enhanced_node.predicate + '-fn.' + verb_pred_label
            
            if bool(predictions[pointer]['need_edge_classify']):
                for edge_index, et in enumerate(edge_targets):
                    if et in seen:
                        continue
                    else:
                        seen.append(et)
                        edge_pred_label = edge_dict_decode[int(predictions[pointer]['edge_pred'][edge_index])]
                        
                        label = [a for a, b in enhanced_node.edges.items() if b == et][0]
                        new_key = label + '-fn.' + edge_pred_label
                        enhanced_node.edges[new_key] = enhanced_node.edges.pop(label)
            pointer += 1
        new_edses.append(enhanced)
    
    for idd, new_eds in zip(ids, new_edses):
        out_dict[idd] = eds_to_string(new_eds)
    with open('../predict_out_gnn.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
        f.close()
    return 

        
if __name__ == '__main__':
    main()