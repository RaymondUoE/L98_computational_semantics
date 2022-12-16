# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch_geometric.nn as geom_nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, gnn_layer=geom_nn.GCNConv, dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        # gnn_layer = layer

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class NodeLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()

        self.model = GNNModel(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index = data.x, data.edge_index
        x = self.model(x, edge_index)
        
        mask = data.mask
        y = torch.nn.functional.one_hot(data.y[mask], num_classes=x[mask].shape[1]).float()
        pred = x[mask].argmax(dim=-1)
        # print(x[mask])
        # print(y)
        loss = self.loss_module(x[mask], y)
        # acc = (pred == data.y[mask]).sum().float() / mask.sum()
        # acc = (pred == data.y[mask]).sum().float() / pred.shape[0]
        # print(acc) return acc
        return loss

    def configure_optimizers(self):
        # We use SGD here, but Adam works as well
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, mode="train")
        self.log('train_loss', loss)
        # self.log('train_acc', acc)
        return loss

    def get_model(self):
        return self.model

    def validation_step(self, batch, batch_idx):
        val_loss = self.forward(batch, mode="val")
        self.log('val_loss', val_loss)

    def test_step(self, batch, batch_idx):
        test_loss = self.forward(batch, mode="test")
        self.log('test_loss', test_loss)