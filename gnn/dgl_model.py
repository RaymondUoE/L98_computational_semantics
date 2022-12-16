import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.nn import GraphConv

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dp_rate=0.1, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dp_rate)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.relu1(h)
        h = self.dropout1(h)
        h = self.conv2(g, h)
        return h


class NodeLevelGNN(pl.LightningModule):

    def __init__(self, batch_size, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        
        self.batch_size = batch_size
        self.save_hyperparameters()

        self.model = GCN(**model_kwargs)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, batched_graph, mode="train"):
        # batched_graph = batched_graph.to('cuda')
        batched_graph = dgl.add_self_loop(batched_graph)
        batched_features = batched_graph.ndata['feat']
        batched_mask = batched_graph.ndata['mask']
        batched_labels = batched_graph.ndata['label'][batched_mask]
        pred = self.model(batched_graph, batched_features)
        pred = pred[batched_mask]
        # x = self.model(x, edge_index)
        # y = torch.nn.functional.one_hot(data.y[mask], num_classes=x[mask].shape[1]).float()
        # pred = x[mask].argmax(dim=-1)
        # print(x[mask])
        # print(y)
        loss = self.loss_module(pred, batched_labels)
        # acc = (pred == data.y[mask]).sum().float() / mask.sum()
        # acc = (pred == data.y[mask]).sum().float() / pred.shape[0]
        # print(acc) return acc
        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=0.001)

        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch, mode="train")
        self.log('train_loss', loss, batch_size=self.batch_size)
        # self.log('train_acc', acc)
        return loss

    def get_model(self):
        return self.model

    def validation_step(self, batch, batch_idx):
        val_loss = self.forward(batch, mode="val")
        self.log('val_loss', val_loss, batch_size=self.batch_size)

    def test_step(self, batch, batch_idx):
        test_loss = self.forward(batch, mode="test")
        self.log('test_loss', test_loss, batch_size=self.batch_size)