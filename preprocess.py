import os
import pandas as pd
import pickle

DEEPBANK_PATH = './deepbank1.1/'
PENN_TREEBANK_PATH = './ptb/nw/wsj'
SEMLINK_FILE = './semlink/other_resources/1.2.2c.okay'
OUT_MAPPINGS_FILE = './sl_mappings.csv'
OUT_SENTENCES_FILE = './sentences.csv'
OUT_TREE_FILE = './trees.csv'

IS_OUTPUT_CSV = True

if not IS_OUTPUT_CSV:
    OUT_MAPPINGS_FILE = './sl_mappings.pkl'
    OUT_SENTENCES_FILE = './sentences.pkl'
    OUT_TREE_FILE = './trees.pkl'

def main():
    print('Preprocessing data...')
    
    process_semlink()
    process_deepbank()
    process_trees()
    
    print('Preprocessing finished.')

        
        
def process_semlink():
    print('processing semlink')
    mappings = []
    with open(SEMLINK_FILE) as f:
        file = f.readlines()
    for line in file:
        contents = line.split()
        sem_map = {}
        sem_map['section_id'] = int(contents[0].split('/')[-2])
        sem_map['doc_id'] = int(contents[0].split('/')[-1].split('_')[1].split('.')[0][-2:])
        sem_map['sentence_id'] = int(contents[1])
        sem_map['token_id'] = int(contents[2])
        sem_map['is_gold'] = contents[3]
        sem_map['vb_form'] = contents[4]
        sem_map['vn_class_index'] = contents[5]
        sem_map['fn_frame'] = contents[6]
        sem_map['pb_sense'] = contents[7]
        sem_map['not_sure_0'] = contents[8]
        sem_map['not_sure_1'] = contents[9]
        sem_map['augmentations'] = contents[10:]

        mappings.append(sem_map)
        
    print('writing file to: '+ OUT_MAPPINGS_FILE)
    if IS_OUTPUT_CSV:
        pd.DataFrame(mappings).sort_values(by=['section_id', 'doc_id', 'sentence_id']).reset_index(drop=True).to_csv(OUT_MAPPINGS_FILE, index=False)
    else:
        with open(OUT_MAPPINGS_FILE, 'wb') as handle:
            pickle.dump(mappings, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    del mappings
        
def process_deepbank():
    print('processing deepbank')
    docs = []

    for folder_name in os.listdir(DEEPBANK_PATH):
        if folder_name[0] != '.':
            for file_name in os.listdir(os.path.join(DEEPBANK_PATH, folder_name)):
                if file_name[0] != '.':
                    # print(file_name)
                    with open(os.path.join(DEEPBANK_PATH, folder_name, file_name), 'r') as f: # open in readonly mode
                        contents = f.read().split('\n\n')
                        doc = {}
                        doc['section_id'] = int(file_name[1:3])
                        doc['doc_id'] = int(file_name[3:5])
                        doc['sentence_id'] = int(file_name[5:]) - 1
                        doc['sentence'] = contents[1].split('`')[1].split('\'')[0]
                        doc['constituency'] = contents[5]
                        doc['eds'] = contents[7]
                        doc['dependency'] = contents[8]

                        docs.append(doc)
    
    print('writing file to: '+ OUT_SENTENCES_FILE)
    if IS_OUTPUT_CSV:
        pd.DataFrame(docs).sort_values(by=['section_id', 'doc_id', 'sentence_id']).reset_index(drop=True).to_csv(OUT_SENTENCES_FILE, index=False)
    else:
        with open(OUT_SENTENCES_FILE, 'wb') as handle:
            pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del docs
            
def process_trees():
    print('processing trees')
    trees = []

    for section_id in os.listdir(PENN_TREEBANK_PATH):
        for file_name in os.listdir(os.path.join(PENN_TREEBANK_PATH, section_id)):
            with open(os.path.join(PENN_TREEBANK_PATH, section_id, file_name), 'r') as f: # open in readonly mode
                contents = f.read().split('\n\n')
                for t, i in zip(contents, range(len(contents))):
                    my_dict = {}
                    my_dict['section_id'] = int(section_id)
                    my_dict['doc_id'] = int(file_name.split('.')[0][-2:])
                    my_dict['sentence_id'] = int(i)
                    my_dict['tree'] = t

                    trees.append(my_dict)
                    
    print('writing file to: '+ OUT_TREE_FILE)
    if IS_OUTPUT_CSV:
        pd.DataFrame(trees).sort_values(by=['section_id', 'doc_id', 'sentence_id']).reset_index(drop=True).to_csv(OUT_TREE_FILE, index=False)    
    else:
        with open(OUT_TREE_FILE, 'wb') as handle:
            pickle.dump(trees, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del trees
            
if __name__ == "__main__":
    main()