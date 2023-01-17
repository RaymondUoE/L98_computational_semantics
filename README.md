# L98_computational_semantics

This is a proejct for L98 Computational Semantics module at the University of Cambridge, MPhil ACS 2022/23 academic year. 


## Data required

Deepbank, semlink (the whole repository) and PTB must be at the current directory.

## Files
### Current directory

 - `cleaner.py`-- Data cleaning
 - `fn_projection.py`-- Part 1, projecting FN frame to EDS. Main script.
 - `predict_out_gnn.json`-- Sample prediction file from GNN.
 - `preprocess.py`-- Preprocessing
 - `train.py`Training pipeline for the classifier module
 - `project_out.json`-- Part 1 output file
### gnn
 - `results` -- results for training, screenshots
 - `dataset.py, gnn_models.py, prepare_data.py, runner.py`-- old skeleton code for PyTorch Geometric. Runner is the runner file.
 - `dgl_....py `-- DGL GNN implementation. Fully working. Runner file is `dgl_runner.py`
 - `featureriser.py`-- Featuriser using BERT
 - `inference.py` -- Inference file, using DGL implementation
 - `utils.py`-- Utilities
 - `featured.py`-- Deprecated
## To run projection:

    python preprocess.py
    python cleaner.py
    python fn_projection.py
 ## To run GNN:
 Assuming the three above commands for projection are run. Then
 

    python cleaner.py -g
   to prepare data for GNN
   

    cd gnn
    python dgl_runner.py
   Models are saved automatically.
   ## To make inference
   

    python inference.py
   Note that model path needs to be changed accordingly. 
    
