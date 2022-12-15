from dataset import EdsDataset
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

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint



DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./model"

def main():
    gnn_layer_by_name = {
        "GCN": geom_nn.GCNConv,
        "GAT": geom_nn.GATConv,
        "GraphConv": geom_nn.GraphConv
    }

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    dataset = EdsDataset('./data', 'gnn_data_small.csv')

def train_node_classifier(model_name, loader, **model_kwargs):
    pl.seed_everything(42)
    node_data_loader = loader

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=200,
                         enable_progress_bar=False) # False because epoch size is 1
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"NodeLevel{model_name}.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = NodeLevelGNN.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything()
        model = NodeLevelGNN(model_name=model_name, c_in=100, c_out=100, **model_kwargs)
        trainer.fit(model, node_data_loader, node_data_loader)
        model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    # test_result = trainer.test(model, node_data_loader, verbose=False)
    # batch = next(iter(node_data_loader))
    # batch = batch.to(model.device)
    # _, train_acc = model.forward(batch, mode="train")
    # _, val_acc = model.forward(batch, mode="val")
    # result = {"train": train_acc,
    #           "val": val_acc,
    #           "test": test_result[0]['test_acc']}
    # return model, result
    return model

if __name__ == "__main__":
    main()