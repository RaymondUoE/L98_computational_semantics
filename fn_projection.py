import pickle
import json
from gnn.utils import *
from tqdm import tqdm

def enhance(edses, semlinks):
    result, enhanced = find_node_ids_edge_targets(edses, semlinks, enhance=True)
    return result, enhanced

if __name__ == "__main__":
    with open('cleaned_data.pkl', 'rb') as file:
        cleaned_data = pickle.load(file)
    file.close()
    out_dict = {}
    
    ids = list(cleaned_data.keys())
    edses = [x['eds'] for x in cleaned_data.values()]
    semlinks = [x['semlink'] for x in cleaned_data.values()]
    node_not_found = 0
    redundant_pb = 0
    for idd, index in tqdm (zip(ids, range(len(ids))), total=len(ids)):
        eds = edses[index]
        semlink = semlinks[index]
        result, enhanced = find_node_ids_edge_targets(eds, semlink, enhance=True)
        out_dict[idd] = eds_to_string(enhanced)
        node_not_found += result['node_cannot_be_found']
        redundant_pb += result['counter_redundant_pb']
        
    print(f'node cannot be found: {node_not_found}')
    print(f'redundant pb roles: {redundant_pb}')
    with open('./projected_out.json', 'w') as f:
        json.dump(out_dict, f, indent=4)
        f.close()