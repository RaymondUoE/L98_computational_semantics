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

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./model"
DATA_FILE_NAME = 'gnn_data_small.csv'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)



def main():
    gnn_layer_by_name = {
        "GCN": geom_nn.GCNConv,
        "GAT": geom_nn.GATConv,
        "GraphConv": geom_nn.GraphConv
    }

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    train_dataset = EdsDataset(root=DATASET_PATH, 
                        filename=DATA_FILE_NAME,
                        mode='train')
    val_dataset = EdsDataset(root=DATASET_PATH, 
                            filename=DATA_FILE_NAME,
                            mode='val')
    test_dataset = EdsDataset(root=DATASET_PATH, 
                            filename=DATA_FILE_NAME,
                            mode='test')
    
    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    graph_val_loader = geom_data.DataLoader(val_dataset, batch_size=64) 
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=64)


    node_gnn_model, node_gnn_result = train_node_classifier(model_name="GNN",
                                                        layer_name="GCN",
                                                        data_loader=graph_train_loader,
                                                        c_hidden=16,
                                                        num_layers=2,
                                                        dp_rate=0.1)
    print_results(node_gnn_result)
def train_node_classifier(model_name, data_loader, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = data_loader
    # node_data_loader = geom_data.DataLoader(dataset, batch_size=1)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         enable_progress_bar=True) # False because epoch size is 1
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = NodeLevelGNN(model_name=model_name, c_in=100, c_out=100, **model_kwargs)
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    test_result = trainer.test(model, node_data_loader, verbose=False)
    batch = next(iter(node_data_loader))
    batch = batch.to(model.device)
    _, train_acc = model.forward(batch, mode="train")
    _, val_acc = model.forward(batch, mode="val")
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_result[0]['test_acc']}
    return model, result
    # return model

def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")

if __name__ == "__main__":
    main()