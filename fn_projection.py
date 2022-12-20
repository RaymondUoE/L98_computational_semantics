import pickle
import json
from gnn.utils import *
from tqdm import tqdm

def enhance(edses, semlinks):
    _, enhanced = find_node_ids_edge_targets(edses, semlinks, enhance=True)
    return enhanced

if __name__ == "__main__":
    with open('cleaned_data.pkl', 'rb') as file:
        cleaned_data = pickle.load(file)
    file.close()
    out_dict = {}
    
    ids = list(cleaned_data.keys())
    edses = [x['eds'] for x in cleaned_data.values()]
    semlinks = [x['semlink'] for x in cleaned_data.values()]
    
    for idd, index in tqdm (zip(ids, range(len(ids))), total=len(ids)):
        eds = edses[index]
        semlink = semlinks[index]
        enhanced = enhance(eds, semlink)
        out_dict[idd] = eds_to_string(enhanced)
        
    with open('./projected_out_new.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
        f.close()