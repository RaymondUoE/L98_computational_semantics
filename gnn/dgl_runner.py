import torch
from dgl_model import GCN, NodeLevelGNN
from dgl_dataset import EdsDataset
import torch.nn.functional as F
import os
import dgl
import pytorch_lightning as pl
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import joblib


CHECKPOINT_PATH = "./model"

MAX_EPOCH = 100
BATCH_SIZE = 32

FEATURE_DIM = 0
CLASS_DIM = 0
EDGE_DIM = 0
torch.manual_seed(22)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def main():
    train_dataset = check_saved_data(mode='train')
    val_dataset = check_saved_data(mode='val')
    test_dataset = check_saved_data(mode='test')
    
    global FEATURE_DIM 
    FEATURE_DIM = train_dataset[0].ndata['feat'].shape[1]
    global CLASS_DIM 
    CLASS_DIM = len(train_dataset.label_dict)
    global EDGE_DIM
    EDGE_DIM = len(train_dataset.edge_dict)
    node_gnn_model, node_gnn_result = train_node_classifier(
                                                        train_dataset=train_dataset,
                                                        val_dataset=val_dataset,
                                                        test_dataset=test_dataset,
                                                        dp_rate=0.1)
    print_results(node_gnn_result)


def train_node_classifier(train_dataset, val_dataset, test_dataset, **model_kwargs):
    model_name = 'GCN'
    pl.seed_everything(67)
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=BATCH_SIZE, drop_last=False)
    val_dataloader = GraphDataLoader(
        val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    test_dataloader = GraphDataLoader(
        test_dataset, batch_size=BATCH_SIZE, drop_last=False)
    
    root_dir = os.path.join(CHECKPOINT_PATH, "NodeLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_accuracy")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=MAX_EPOCH,
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    model = NodeLevelGNN(model_name=model_name, 
                         batch_size=BATCH_SIZE, 
                         in_feats=FEATURE_DIM,
                         h_feats=728,
                         d_embed=728, 
                         verb_class=CLASS_DIM,
                         edge_class=EDGE_DIM,
                        #  num_classes=CLASS_DIM, 
                         **model_kwargs)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    train_result = trainer.test(model, dataloaders=train_dataloader)[0]
    val_result = trainer.test(model, dataloaders=val_dataloader)[0]
    test_result = trainer.test(model, dataloaders=test_dataloader)[0]
    result = {"train": train_result['test_accuracy'],
              "val": val_result['test_accuracy'],
              "test": test_result['test_accuracy']}
    return model, result

def pl_predict(model, predict_dataloader):
        torch.set_grad_enabled(False)
        model.eval()
        verb_pred_all = []
        edge_pred_all = []

        for batch_idx, batch in enumerate(predict_dataloader):
            result = model.predict_step(batch, batch_idx)
            verb_pred_all.append(result['verb_pred'])
            if result['need_edge_classify']:
                edge_pred_all.append(result['edge_pred'])
            else:
                edge_pred_all.append(-1)
        return verb_pred_all, edge_pred_all
        
def check_saved_data(mode):
    if mode not in ['train', 'val', 'test']:
        raise Exception('Mode error. train val test')
    if os.path.exists(f'./data/processed/dgl_graphs_{mode}.pkl'):
        print(f'Data found, mode = {mode}. Loading...')
        with open(f'./data/processed/dgl_graphs_{mode}.pkl', 'rb') as f:
            dataset = joblib.load(f)
            f.close()
    else:
        dataset = EdsDataset(mode=mode)
        with open(f'./data/processed/dgl_graphs_{mode}.pkl', 'wb') as f:
            joblib.dump(dataset, f)
            f.close()
    return dataset
    




def print_results(result_dict):
    if "train" in result_dict:
        print(f"Train accuracy: {(100.0*result_dict['train']):4.2f}%")
    if "val" in result_dict:
        print(f"Val accuracy:   {(100.0*result_dict['val']):4.2f}%")
    print(f"Test accuracy:  {(100.0*result_dict['test']):4.2f}%")

def predict(gcn, verb_classify, edge_classify, dataset):
    num_correct = 0
    for batched_graph in dataset:
        # batched_graph = dgl.add_self_loop(batched_graph)
        batched_features = batched_graph.ndata['feat']
        batched_verb_mask = batched_graph.ndata['verb_mask']
        batched_arg_mask = batched_graph.ndata['edge_mask']
        batched_verb_labels = batched_graph.ndata['verb_label'][batched_verb_mask]
        batched_edge_labels = batched_graph.ndata['edge_label'][batched_arg_mask]
        node_embed = gcn(batched_graph, batched_features)
        verb_embed = node_embed[batched_verb_mask]
        arg_embed = node_embed[batched_arg_mask]
        verb_num_of_children = batched_graph.ndata['verb_num_children'][batched_verb_mask]
        
        v_stack = []
        for i, num in zip(range(verb_num_of_children.shape[0]), verb_num_of_children):
            v = verb_embed[i, :]
            if num > 0:
                v_stack.append(v.unsqueeze(0).repeat(num,1))
        v_stack_emb = torch.cat(v_stack, dim=0).to(self.device)
        
        verb_class = verb_classify(verb_embed)
        arg_class = edge_classify(v_stack_emb, arg_embed)
        
        batched_verb_labels = batched_graph.ndata['verb_label'][batched_verb_mask]
        batched_edge_labels = batched_graph.ndata['edge_label'][batched_arg_mask]
        
        pred = pred.argmax(dim=-1)
        num_correct += (pred == batched_labels).sum().item()
        
    return float(num_correct)/len(dataset)

if __name__ == "__main__":
    main()
    # old()
