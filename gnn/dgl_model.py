import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from dgl.nn import GraphConv

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, d_embed, dp_rate=0.1, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.relu1= nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dp_rate)
        self.conv2 = GraphConv(h_feats, d_embed)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = self.relu1(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h

class VerbClassifyStack(nn.Module):
    def __init__(self, d_embed, verb_class, dp_rate=0.1):
        super(VerbClassifyStack, self).__init__()
        self.flatten = nn.Flatten()
        self.verb_classify_stack = nn.Sequential(
            nn.Linear(d_embed, 512),
            nn.ReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, verb_class),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.verb_classify_stack(x)
        return logits
    
class EdgeClassifyStack(nn.Module):
    def __init__(self, d_embed, edge_class, dp_rate=0.1):
        super(EdgeClassifyStack, self).__init__()
        self.flatten = nn.Flatten()
        self.edge_classify_stack = nn.Sequential(
            nn.Linear(3*d_embed, 768),
            nn.ReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dp_rate),
            nn.Linear(512, edge_class),
        )

    def forward(self, vb, arg):
        vb = self.flatten(vb)
        arg = self.flatten(arg)
        logits = self.edge_classify_stack(torch.cat([vb, arg, torch.abs(vb - arg)], dim=1))
        return logits
    
class NodeLevelGNN(pl.LightningModule):

    def __init__(self, batch_size, in_feats, h_feats, d_embed, verb_class, edge_class, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.d_embed = d_embed
        self.model = GCN(in_feats, h_feats, d_embed)
        self.verb_classify = VerbClassifyStack(d_embed, verb_class)
        self.edge_classify = EdgeClassifyStack(d_embed, edge_class)
        
        self.loss_module = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(verb_class+edge_class))
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(verb_class+edge_class))
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=(verb_class+edge_class))
        
    def forward(self, batched_graph, mode='train'):
        batched_features = batched_graph.ndata['feat']
        batched_verb_mask = batched_graph.ndata['verb_mask']
        batched_arg_mask = batched_graph.ndata['edge_mask']
        
        node_embed = self.model(batched_graph, batched_features)
        verb_embed = node_embed[batched_verb_mask]
        arg_embed = node_embed[batched_arg_mask]
        verb_num_of_children = batched_graph.ndata['verb_num_children'][batched_verb_mask]
        need_edge_classify = torch.sum(verb_num_of_children) != 0
        if need_edge_classify:
            v_stack = []
            for i, num in zip(range(verb_num_of_children.shape[0]), verb_num_of_children):
                v = verb_embed[i, :]
                if num > 0:
                    v_stack.append(v.unsqueeze(0).repeat(num,1))
            v_stack_emb = torch.cat(v_stack, dim=0).to(self.device)
            arg_class = self.edge_classify(v_stack_emb, arg_embed)
            if mode != 'pred':
                batched_edge_labels = batched_graph.ndata['edge_label'][batched_arg_mask]

        verb_class = self.verb_classify(verb_embed)
        if mode != 'pred':
            batched_verb_labels = batched_graph.ndata['verb_label'][batched_verb_mask]

            loss_verb = self.loss_module(verb_class, batched_verb_labels)
            loss_edge = self.loss_module(arg_class, batched_edge_labels) if need_edge_classify else 0
            loss = loss_verb + loss_edge
            return {'loss':loss,
                    'verb_class': verb_class,
                    'arg_class':arg_class if need_edge_classify else None,
                    'batched_verb_labels':batched_verb_labels,
                    'batched_edge_labels':batched_edge_labels if need_edge_classify else None,
                    'need_edge_classify':need_edge_classify
                    }
        else:
            return {
                    'verb_class': verb_class,
                    'arg_class':arg_class if need_edge_classify else None,
                    'need_edge_classify':need_edge_classify
                    }
             
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=2e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        result = self.forward(batch, mode="train")
        train_loss = result['loss']
        verb_class = result['verb_class']
        arg_class = result['arg_class']
        batched_verb_labels = result['batched_verb_labels']
        batched_edge_labels = result['batched_edge_labels']
        need_edge_classify = result['need_edge_classify']
        verb_pred = verb_class.argmax(dim=-1)
        if need_edge_classify:
            edge_pred = arg_class.argmax(dim=-1)
            self.train_acc.update(edge_pred, batched_edge_labels)
        self.train_acc.update(verb_pred, batched_verb_labels)
        self.log("train_loss", train_loss, batch_size=self.batch_size)
        return train_loss
    
    def training_epoch_end(self, training_step_outputs):
        train_accuracy = self.train_acc.compute()
        self.log("train_accuracy", train_accuracy, batch_size=self.batch_size)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch, mode="val")
        val_loss = result['loss']
        verb_class = result['verb_class']
        arg_class = result['arg_class']
        batched_verb_labels = result['batched_verb_labels']
        batched_edge_labels = result['batched_edge_labels']
        need_edge_classify = result['need_edge_classify']
        verb_pred = verb_class.argmax(dim=-1)
        if need_edge_classify:
            edge_pred = arg_class.argmax(dim=-1)
            self.val_acc.update(edge_pred, batched_edge_labels)
        self.val_acc.update(verb_pred, batched_verb_labels)
        
        self.log("val_loss", val_loss, batch_size=self.batch_size)
        return val_loss

    def validation_epoch_end(self, validation_step_outputs):
        val_accuracy = self.val_acc.compute()
        self.log("val_accuracy", val_accuracy, batch_size=self.batch_size)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        result = self.forward(batch, mode="val")
        test_loss = result['loss']
        verb_class = result['verb_class']
        arg_class = result['arg_class']
        batched_verb_labels = result['batched_verb_labels']
        batched_edge_labels = result['batched_edge_labels']
        need_edge_classify = result['need_edge_classify']
        verb_pred = verb_class.argmax(dim=-1)
        if need_edge_classify:
            edge_pred = arg_class.argmax(dim=-1)
            self.test_acc.update(edge_pred, batched_edge_labels)
        self.test_acc.update(verb_pred, batched_verb_labels)
        
        self.log("test_loss", test_loss, batch_size=self.batch_size)
        return test_loss
    
    def test_epoch_end(self, test_step_outputs):
        test_accuracy = self.test_acc.compute()
        self.log("test_accuracy", test_accuracy, batch_size=self.batch_size)
        self.test_acc.reset()
    
    def predict_step(self, batch, batch_idx):
        result = self.forward(batch, mode='pred')
        verb_class = result['verb_class']
        arg_class = result['arg_class']
        need_edge_classify = result['need_edge_classify']
        verb_pred = verb_class.argmax(dim=-1)
        if need_edge_classify:
            edge_pred = arg_class.argmax(dim=-1)
        return {'verb_pred': verb_pred,
                'edge_pred': edge_pred if need_edge_classify else None,
                'need_edge_classify': need_edge_classify}
        
    def get_model(self):
        return self.model, self.verb_classify, self.edge_classify