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

MAX_EPOCH = 50
BATCH_SIZE = 32

FEATURE_DIM = 0
CLASS_DIM = 0


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

def main():
    # num_examples = len(dataset)
    # num_train = int(num_examples * 0.8)
    

    # train_sampler = SubsetRandomSampler(torch.arange(num_train))
    # test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))
    train_dataset = check_saved_data(mode='train')
    val_dataset = check_saved_data(mode='val')
    test_dataset = check_saved_data(mode='test')

    global FEATURE_DIM 
    FEATURE_DIM = train_dataset[0].ndata['feat'].shape[1]
    global CLASS_DIM 
    CLASS_DIM = len(train_dataset.label_dict)
    node_gnn_model, node_gnn_result = train_node_classifier(
                                                        train_dataset=train_dataset,
                                                        val_dataset=val_dataset,
                                                        test_dataset=test_dataset,
                                                        h_feats=16,
                                                        num_layers=2,
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
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_loss")],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=MAX_EPOCH,
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    model = NodeLevelGNN(model_name=model_name, batch_size=BATCH_SIZE, in_feats=FEATURE_DIM, num_classes=CLASS_DIM, **model_kwargs)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # best_model = os.path.join(trainer.checkpoint_callback.best_model_path, f"NodeLevel{model_name}.ckpt")
    model = NodeLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on the test set
    model_used_to_predict = model.get_model()
   
    train_acc = predict(model_used_to_predict, train_dataset)
    val_acc = predict(model_used_to_predict, val_dataset)
    test_acc = predict(model_used_to_predict, test_dataset)
    result = {"train": train_acc,
              "val": val_acc,
              "test": test_acc}
    return model, result
    # return model


def old():
    train_dataset = check_saved_data(mode='train')
    val_dataset = check_saved_data(mode='val')
    test_dataset = check_saved_data(mode='test')
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=BATCH_SIZE, drop_last=False)
    val_dataloader = GraphDataLoader(
        val_dataset, batch_size=BATCH_SIZE, drop_last=False)
    test_dataloader = GraphDataLoader(
        test_dataset, batch_size=BATCH_SIZE, drop_last=False)

    model = GCN(train_dataset[0].ndata['feat'].shape[1], 16, len(train_dataset.label_dict)).to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        for batched_graph in train_dataloader:
            batched_graph = batched_graph.to('cuda')
            batched_graph = dgl.add_self_loop(batched_graph)
            batched_features = batched_graph.ndata['feat']
            batched_mask = batched_graph.ndata['mask']
            
            batched_labels = batched_graph.ndata['label'][batched_mask]
            pred = model(batched_graph, batched_features)[batched_mask]
            loss = F.cross_entropy(pred, batched_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    num_correct = 0
    num_tests = len(val_dataset)
    for batched_graph in val_dataloader:
        batched_graph = batched_graph.to('cuda')
        batched_graph = dgl.add_self_loop(batched_graph)
        batched_features = batched_graph.ndata['feat']
        batched_mask = batched_graph.ndata['mask']

        batched_labels = batched_graph.ndata['label'][batched_mask]
        pred = model(batched_graph, batched_features)[batched_mask]
        pred = pred.argmax(dim=-1)
        num_correct += (pred == batched_labels).sum().item()
        # num_tests += len(labels)
    print(float(num_correct)/num_tests)
    # g = train_dataset[0]
    # g = g.to('cuda')
    # model = GCN(g.ndata['feat'].shape[1], 16, len(train_dataset.label_dict)).to('cuda')

    # train(g, model)


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

def predict(model, dataset):
    num_correct = 0
    for batched_graph in dataset:
        # print(batched_graph)
        # num_of_tests += len(batched_graph)
        # pred = model.forward(d.x, d.edge_index)
        batched_graph = dgl.add_self_loop(batched_graph)
        batched_features = batched_graph.ndata['feat']
        batched_mask = batched_graph.ndata['mask']
        batched_labels = batched_graph.ndata['label'][batched_mask]
        pred = model.forward(batched_graph, batched_features)[batched_mask]
        # pred = pred[batched_mask]


        # pred = model.forward(d.x, d.edge_index)
        # pred = pred[batched_mask].argmax(dim=-1)
        # y = d.y[d.mask]
        pred = pred.argmax(dim=-1)
        num_correct += (pred == batched_labels).sum().item()
        
    return float(num_correct)/len(dataset)

if __name__ == "__main__":
    main()
    # old()
