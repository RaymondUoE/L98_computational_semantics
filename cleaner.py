import sys, getopt
import pickle
import delphin.codecs.eds

import pandas as pd

from sklearn.utils import shuffle
from utils import *

OUT_MAPPINGS_FILE = './sl_mappings.csv'
OUT_SENTENCES_FILE = './sentences.csv'
OUT_TREE_FILE = './trees.csv'

# FILTER_EDS = True
# FILTER_SEMLINK = False
# FILTER_TREE = False

PREPARE_FOR_GNN = False


def main(argv):

    opts, args = getopt.getopt(argv,"hg",["gnn-only"])
    for opt, arg in opts:
        if opt == '-h':
            print ('cleaner.py (-g)')
            sys.exit()
        else:
            PREPARE_FOR_GNN = opt in ("-g", "--gnn-only")
        
    if PREPARE_FOR_GNN:
        print('Processing data for GNN...')

        prepare_gnn()
    else:
        print('Cleaning data...')

        clean_data()

def build_node_target_dict(dicts):
    labels = []
    for d in dicts:
        labels.append(d['fn_frame'])
    labels = list(set(labels))

    node_target_dict = {}
    for l, index in zip(labels, range(len(labels))):
        node_target_dict[l] = index
        # TODO

def prepare_gnn():
    # pass
    print('Loading dataset...')

    with open('cleaned_data.pickle', 'rb') as file:
        cleaned_data = pickle.load(file)
    file.close()

    dicts = []
    counter_redundant_pb = 0
    node_cannot_be_found = 0


    for k, v in cleaned_data.items():
        # it has semlink
        if 'semlink' in v and 'eds' in v:
            result = find_node_ids_edge_targets(v['eds'], v['semlink'])
            counter_redundant_pb += result['counter_redundant_pb']
            node_cannot_be_found += result['node_cannot_be_found']
            
            for i in range(len(result['node_ids'])):
                temp_dict = {}
                temp_dict['id'] = k
                temp_dict['sentence'] = v['sentence']
                temp_dict['eds'] = delphin.codecs.eds.encode(v['eds'])
                temp_dict['semlink'] = result['semlink'][i]
                temp_dict['target_node'] = result['node_ids'][i]
                temp_dict['fn_frame'] = result['fn_frames'][i]
                temp_dict['edge_targets'] = result['edge_targets'][i]
                temp_dict['fn_roles'] = result['fn_roles'][i]
                dicts.append(temp_dict)
        else:
            pass
    print('PB role repeats in semlink: {} times'.format(counter_redundant_pb))
    print('EDS does not have corresponding node for semlink {} times'.format(node_cannot_be_found))
    print('Clean data points: {}'.format(len(dicts)))
    df = pd.DataFrame(dicts)
    df = shuffle(df, random_state=100)
    df.to_csv('gnn_data.csv',index=False)

    build_node_target_dict(dicts)


def clean_data():

    print('Loading data...')
    sentences = pd.read_csv('sentences.csv')
    semlink_map = pd.read_csv('sl_mappings.csv')
    trees = pd.read_csv('trees.csv')

    print('Cleaning data...')

    cleaned_data = {}
    eds_missing = []
    eds_failure = []
    semlink_failure = []
    semlink_missing = []
    tree_failure = []
    tree_missing = []

    # find missing eds
    print('Finding missing EDS...')
    for index, row in semlink_map.iterrows():
        section_id = row['section_id']
        doc_id = row['doc_id']
        sentence_id = row['sentence_id']
        all_index = str(section_id).zfill(3) + str(doc_id).zfill(3) + str(sentence_id).zfill(3)

        if not find_eds_by_ids_df(section_id, doc_id, sentence_id, sentences):
            eds_missing.append(all_index)
    
    eds_missing = list(set(eds_missing))



    for index, row in sentences.iterrows():
        temp_dict = {}
        section_id = row['section_id']
        doc_id = row['doc_id']
        sentence_id = row['sentence_id']
        all_index = str(section_id).zfill(3) + str(doc_id).zfill(3) + str(sentence_id).zfill(3)

        # filter EDS
        try:
            cur_eds = eds_from_string(row['eds'])
            temp_dict['sentence'] = row['sentence']
            temp_dict['eds'] = cur_eds
        except:
            eds_failure.append(all_index)

        # find semlink:
        try:
            semlink_result = find_semlink_by_ids_df(section_id,doc_id,sentence_id,semlink_map)
            if semlink_result:
                temp_dict['semlink'] = semlink_result
            else:
                semlink_missing.append(all_index)
        except:
            semlink_failure.append(all_index)

        # find tree
        try:
            tree_result = find_tree_by_ids_df(section_id,doc_id,sentence_id,trees)
            if tree_result:
                temp_dict['tree'] = tree_result
            else:
                tree_missing.append(all_index)
        except:
            tree_failure.append(all_index)
        
        cleaned_data[all_index] = temp_dict

    error_record = {}
    error_record['eds_missing'] = eds_missing
    error_record['semlink_missing'] = semlink_missing
    error_record['tree_missing'] = tree_missing
    error_record['eds_failure'] = eds_failure
    error_record['semlink_failure'] = semlink_failure
    error_record['tree_failure'] = tree_failure


    print('Cleaning complete...')
    print('Cleaned records: ' + str(len(cleaned_data)))
    print('EDS missing: ' + str(len(eds_missing)))
    print('Semlinks missing: ' + str(len(semlink_missing)))
    print('Tree missing: ' + str(len(tree_missing)))
    print('EDS failure: ' + str(len(eds_failure)))
    print('Semlinks failure: ' + str(len(semlink_failure)))
    print('Tree failure: ' + str(len(tree_failure)))

    print('Saving...')
    
    with open('cleaned_data.pickle', 'wb') as handle:
        pickle.dump(cleaned_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()
    with open('error_record.pickle', 'wb') as handle:
        pickle.dump(error_record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    del sentences
    del semlink_map 
    del trees
    del cleaned_data 
    del eds_missing 
    del eds_failure 
    del semlink_failure
    del semlink_missing
    del tree_failure
    del tree_missing

if __name__ == "__main__":
    main(sys.argv[1:])