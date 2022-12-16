import pickle
import json
import joblib
import torch

with open('../../code/frame_net_project/verb_transformed.pkl', 'rb') as f:
    data = joblib.load(f)
    f.close()

num_of_data = len(data)
val_index = int(num_of_data * 0.8)
test_index = int(num_of_data * 0.9)



print('Processing data...')
train_data = data[:val_index]
torch.save(train_data, '../../code/frame_net_project/data/processed/processed_data_train.pt')
del train_data

val_data = data[val_index:test_index]
torch.save(val_data, '../../code/frame_net_project/data/processed/processed_data_val.pt')
del val_data
test_data = data[test_index:]
torch.save(test_data, '../../code/frame_net_project/data/processed/processed_data_test.pt')
del test_data