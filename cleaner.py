import sys, getopt
import pickle
# import joblib
import delphin.codecs.eds

import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from tqdm import tqdm
from gnn.utils import *


def main(argv):
    PREPARE_FOR_GNN = False

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


def prepare_gnn():
    print('Loading dataset...')

    with open('cleaned_data.pkl', 'rb') as file:
        cleaned_data = pickle.load(file)
    file.close()

    dicts = []
    counter_redundant_pb = 0
    node_cannot_be_found = 0


    for k, v in cleaned_data.items():
        # it has semlink
        if 'semlink' in v and 'eds' in v:
            result, _ = find_node_ids_edge_targets(v['eds'], v['semlink'])
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
    df = df.dropna(how='any', axis=0)
    df = shuffle(df, random_state=100)
    # subsampling missing data
    df['all_children_labelled'] = list(map(
                                        lambda x: x > 0, map(
                                            lambda x: np.product([1 if y != '' else 0 for y in x]), df['fn_roles'])))
    df_filter_frame = df[~df['fn_frame'].isin(['NF','IN'])]
    all_labelled = df_filter_frame[(df_filter_frame['all_children_labelled'] == True) | (df_filter_frame['edge_targets'] == '[]')]
    only_children_missing = df_filter_frame[(df_filter_frame['all_children_labelled'] == False) & (df_filter_frame['edge_targets'] != '[]')]
    all_missing = df[(df['fn_frame'].isin(['NF','IN'])) & (df['all_children_labelled'] == False) & (df['edge_targets'] != '[]')]
    add_num = int(len(all_labelled) / 0.98 * 0.01)
    adding_children_missing = only_children_missing.sample(frac=1).reset_index(drop=True)[:add_num]
    adding_all_missing = all_missing.sample(frac=1).reset_index(drop=True)[:add_num]
    df = pd.concat([all_labelled, adding_children_missing, adding_all_missing], ignore_index=True).sample(frac=1).reset_index(drop=True)
    print(f'All labeled data: {len(all_labelled)}')
    print(f'Only edge missing: {len(adding_children_missing)}')
    print(f'All labels missing: {len(adding_all_missing)}')
    num_of_data = len(df)
    val_index = int(num_of_data * 0.8)
    test_index = int(num_of_data * 0.9)

    print('Spliting data...')
    train_data = df[:val_index]
    val_data = df[val_index:test_index]
    test_data = df[test_index:]
    train_data.to_csv('./gnn/data/raw/gnn_data_dgl_train.csv',index=False)
    val_data.to_csv('./gnn/data/raw/gnn_data_dgl_val.csv',index=False)
    test_data.to_csv('./gnn/data/raw/gnn_data_dgl_test.csv',index=False)
    # df.to_csv('gnn_data.csv',index=False)

    # build_node_target_dict(dicts)


def clean_data():

    print('Loading data...')
    sentences = pd.read_csv('sentences.csv')
    semlink_map = pd.read_csv('sl_mappings.csv')
    trees = pd.read_csv('trees.csv')

    print(f'Loaded {len(sentences)} EDSes,\n{len(semlink_map)} semlink entries,\n{len(trees)} constituency trees.')
    print('Cleaning data...')

    cleaned_data = {}
    eds_missing = []
    eds_failure = []
    semlink_failure = []
    semlink_missing = []
    tree_failure = []
    tree_missing = []

    # find missing eds - ids in semlink but not in eds
    print('Finding missing EDS...')
    all_df = semlink_map.drop_duplicates(subset=['id']).merge(sentences.drop_duplicates(), on=['id'], how='left', indicator=True)
    eds_missing_df = all_df[all_df['_merge'] == 'left_only']
    del all_df
    eds_missing = list(eds_missing_df['id'])
    del eds_missing_df
    
    all_df = semlink_map.merge(sentences, on=['id'], how='outer', indicator=True)
    eds_missing = list(all_df[all_df['_merge'] == 'left_only'].drop_duplicates(subset=['id'])['id'])
    semlink_missing = list(all_df[all_df['_merge'] == 'right_only'].drop_duplicates(subset=['id'])['id'])
    
    # filtered_ids = [x for x in list(sentences['id']) if not x in semlink_missing]
    
    filtered_sentences = sentences[~sentences.id.isin(semlink_missing)]


    # for index, row in tqdm(sentences.iterrows(), total=len(sentences)):
    for index, row in tqdm(filtered_sentences.iterrows(), total=len(filtered_sentences)):
        temp_dict = {}
        # section_id = row['section_id']
        # doc_id = row['doc_id']
        # sentence_id = row['sentence_id']
        # all_index = str(section_id).zfill(3) + str(doc_id).zfill(3) + str(sentence_id).zfill(3)
        cur_id = row['id']

        # filter EDS
        try:
            cur_eds = eds_from_string(row['eds'])
            temp_dict['sentence'] = row['sentence']
            temp_dict['eds'] = cur_eds
        except:
            eds_failure.append(cur_id)
            # don't add these ids in cleaned datasets
            continue

        # find semlink:
        try:
            semlink_result = find_df_by_id(cur_id ,semlink_map)
            if semlink_result:
                temp_dict['semlink'] = semlink_result

        except:
            semlink_failure.append(cur_id)
            continue

        # find tree
        try:
            tree_result = find_tree_by_ids_df(cur_id, trees)
            if tree_result:
                temp_dict['tree'] = tree_result
            else:
                tree_missing.append(cur_id)
        except:
            tree_failure.append(cur_id)
        
        cleaned_data[cur_id] = temp_dict

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
    
    with open('cleaned_data.pkl', 'wb') as handle:
        pickle.dump(cleaned_data, handle)
        handle.close()
    with open('error_record.pkl', 'wb') as handle:
        pickle.dump(error_record, handle)
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