import pickle
import delphin.codecs.eds

import pandas as pd

OUT_MAPPINGS_FILE = './sl_mappings.csv'
OUT_SENTENCES_FILE = './sentences.csv'
OUT_TREE_FILE = './trees.csv'

FILTER_EDS = True
FILTER_SEMLINK = False
FILTER_TREE = False


def main():
    print('Loading data...')
    sentences = pd.read_csv('sentences.csv')
    semlink_map = pd.read_csv('sl_mappings.csv')
    trees = pd.read_csv('trees.csv')

    print('Cleaning data...')

    for index, row in sentences.iterrows():
        section_id = row['section_id']
        doc_id = row['doc_id']
        sentence_id = row['sentence_id']
        section_id.zfill()





























if __name__ == "__main__":
    main()