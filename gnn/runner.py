from dataset import EdsDataset
from gnn_models import GNNModel, NodeLevelGNN
import os
import json
import math
import numpy as np
import time



# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
# import torch_geometric.loader as loader
from torch_geometric.loader import DataLoader

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./model"
DATA_FILE_NAME = 'gnn_data_small.csv'
MAX_EPOCH = 5
BATCH_SIZE = 1

FEATURE_DIM = 0
CLASS_DIM = 0

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

with open('node_label_dict.json', 'r') as f:
    node_label_dict = json.load(f)
    f.close()


def main():
    gnn_layer_by_name = {
        "GCN": geom_nn.GCNConv,
        "GAT": geom_nn.GATConv,
        "GraphConv": geom_nn.GraphConv
    }

    # train_dataset = EdsDataset(root=DATASET_PATH, 
    #                     filename=DATA_FILE_NAME,
    #                     mode='train')
    # val_dataset = EdsDataset(root=DATASET_PATH, 
    #                         filename=DATA_FILE_NAME,
    #                         mode='val')
    # test_dataset = EdsDataset(root=DATASET_PATH, 
    #                         filename=DATA_FILE_NAME,
    #                         mode='test')
    


    # graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    # graph_val_loader = geom_data.DataLoader(val_dataset, batch_size=1) 
    # graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=1)
    # print('----------------')
    # print(train_dataset[0])

    train_datalist = torch.load('./data/processed/processed_data_train.pt')
    val_datalist = torch.load('./data/processed/processed_data_val.pt')
    test_datalist = torch.load('./data/processed/processed_data_test.pt')

    # train_datalist = torch.load('../../code/frame_net_project/data/processed/processed_data_train.pt')
    # val_datalist = torch.load('../../code/frame_net_project/data/processed/processed_data_val.pt')
    # test_datalist = torch.load('../../code/frame_net_project/data/processed/processed_data_test.pt')


    global FEATURE_DIM 
    FEATURE_DIM = train_datalist[0].x.shape[1]
    global CLASS_DIM 
    CLASS_DIM = len(node_label_dict)
    # print('-------------')
    # print(train_datalist[:3])
    node_gnn_model, node_gnn_result = train_node_classifier(model_name='GCN',
                                                        train_dataset=train_datalist,
                                                        val_dataset=val_datalist,
                                                        test_dataset=test_datalist,
                                                        c_hidden=16,
                                                        num_layers=2,
                                                        dp_rate=0.1)
    print_results(node_gnn_result)

def train_node_classifier(model_name, train_dataset, val_dataset, test_dataset, **model_kwargs):
    pl.seed_everything(67)
    train_data_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) 
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    # print('----------------')
    # print(train_data_loader)
    # train_data_loader = geom_data.DataLoader(train_datalist, batch_size=1)
    # val_data_loader = geom_data.DataLoader(val_datalist, batch_size=1) 
    # test_data_loader = geom_data.DataLoader(test_datalist, batch_size=1)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_loss")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=MAX_EPOCH,
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(67)
        model = NodeLevelGNN(model_name=model_name, c_in=FEATURE_DIM, c_out=CLASS_DIM, **model_kwargs)
        trainer.fit(model, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)
        # best_model = os.path.join(trainer.checkpoint_callback.best_model_path, f"NodeLevel{model_name}.ckpt")
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    model_used_to_predict = model.get_model()
    # Test best model on validation and test set
    # train_result = trainer.test(model, graph_train_loader, verbose=False)
    # test_result = trainer.test(model, graph_test_loader, verbose=False)
    # result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
    # train_result = trainer.test(model, train_data_loader, verbose=False)
    # val_result = trainer.test(model, val_data_loader, verbose=False)
    # test_result = trainer.test(model, test_data_loader, verbose=False)
    # result = {"test": test_result[0]['test_acc'], "train": train_result[0]['test_acc']}
    train_acc = predict(model_used_to_predict, train_dataset)
    val_acc = predict(model_used_to_predict, val_dataset)
    test_acc = predict(model_used_to_predict, test_dataset)
  
    # test_result = trainer.test(model, test_data_loader, verbose=False)
    # train_batch = next(iter(train_data_loader)).to(model.device)
    # val_batch = next(iter(val_data_loader)).to(model.device)
    # train_loss = model.forward(train_batch, mode="train")
    # val_loss= model.forward(val_batch, mode="val")
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_acc}
    return model, result
    # return model

def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")

def predict(model, dataset):
    counter = 0
    for d in dataset:
        pred = model.forward(d.x, d.edge_index)
        pred = pred[d.mask].argmax(dim=-1)
        y = d.y[d.mask]
        if y.item() == pred.item():
            counter += 1

    return float(counter)/len(dataset)

if __name__ == "__main__":
    main()