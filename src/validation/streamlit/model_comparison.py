import os
import sys
import torch
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import vis
import st_utils

### Locus Selection ###
chr_num, start = st_utils.locus_config(st, chr_idx = 14, start_init = 59100000)
loci_info = [chr_num, str(start)]

### Model Selection ###
model_data_preprocessing_reference = 'strawberry'
model_names = ['strawberry', 'pineapple']
#model_names = ['strawberry']
#model_names = ['pineapple']



def load_strawberry():
    sys.path.append('/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/C.Origami/src/training') # New model folder
    import main
    pl_module = main.TrainModule.load_from_checkpoint('/gpfs/home/jt3545/projects/C.Origami/src/training/runs/c-origami-baseline-greene/2022-06-26/04-01-39-420100/models/epoch=82-step=49384.ckpt')
    sys.path.remove('/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/C.Origami/src/training') # New model folder
    pl_module.eval()
    return pl_module

@st.cache(allow_output_mutation=True)
def get_models(model_names):
    loaded_models = {}
    for model_name in model_names:
        # Get the new model
        if model_name == 'strawberry':
            model = load_strawberry()
            loaded_models[model_name] = model
            print('strawberry loaded')

        if model_name == 'pineapple':
            model_st = 'pineapple'
            model_cell_st = 'IMR90' 
            epoch_st = '49'
            model_dir = st_utils.get_model_path(model_st, model_cell_st, epoch_st)
            sys.path.append('/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-visualization/src') # Original Test folder
            import importlib
            model_kit = importlib.import_module(f'models.pineapple.model_kit')
            model = model_kit.CustomModel(model_dir)
            loaded_models[model_name] = model
            print(model.model_id())
    return loaded_models
loaded_models = get_models(model_names)

### Cell Type Selection ###
sys.path.append('/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-visualization/src') # Original Test folder
from models.model_class import CellType 

cell_type_names = ['imr90', 'gm12878']
num_cell_types = len(cell_type_names)

# Get corresponding CTCF and ATAC
@st.cache(allow_output_mutation=True)
def get_cells(cell_type_names, ctcf_name, atac_name):
    cell_types = {}
    for cell_type_name in cell_type_names:
        cell_types[cell_type_name] = CellType(st_utils.get_cell_path(cell_type_name), ctcf_name, atac_name)
    return cell_types

ctcf_name, atac_name = st_utils.get_ctcf_atac_names(model_data_preprocessing_reference)
cell_types = get_cells(cell_type_names, ctcf_name, atac_name)

### Compute Prediction ###

def onehot_encode(seq):
    ''' 
    encode integer dna array to onehot (n x 5)
    Args:
        seq (arr): Numpy array (n x 1) of dna encoded as 0-4 integers

    Returns:
        array: A numpy matrix (n x 5)
    '''
    seq_emb = np.zeros((len(seq), 5))
    seq_emb[np.arange(len(seq)), seq] = 1
    return seq_emb

def predict_strawberry(inputs, model):
    seq, ctcf, atac, mat = inputs
    # proc_seq
    seq = torch.tensor(onehot_encode(seq)).unsqueeze(0)
    # normailze input
    ctcf = torch.tensor(np.nan_to_num(ctcf, 0)) # Important! replace nan with 0
    atac_log = torch.tensor(np.log(np.nan_to_num(atac, 0) + 1)) # Important! replace nan with 0
    features = [ctcf, atac_log]

    features = torch.cat([feat.unsqueeze(0).unsqueeze(2) for feat in features], dim = 2)
    inputs = torch.cat([seq, features], dim = 2).cuda()

    model.cuda()
    model.model.record_attn = True
    model.model.attn.module.record_attn = True
    outputs, attn_weights = model(inputs)
    ### Need to modify torch/nn/functional.py at line 5089 to make this work: 
    # return attn_output, attn_output_weights#.sum(dim=1) / num_heads
    #import pdb; pdb.set_trace()
    print(attn_weights.shape)
    outputs = outputs[0].detach().cpu().numpy()
    np.save('attn_weights.npy', attn_weights.detach().cpu().numpy())
    return outputs

cell_data = {}
models_preds = {}
for cell_type_name in cell_type_names:
    pred_data = {}
    cell_data[cell_type_name] = cell_types[cell_type_name].get_data(chr_num, start)
    for model_name in model_names:
        if model_name == 'strawberry':
            pred_data[model_name] = predict_strawberry(cell_data[cell_type_name], loaded_models[model_name])
        else:
            pred_data[model_name], _ = loaded_models[model_name].predict(*cell_data[cell_type_name])
    models_preds[cell_type_name] = pred_data

### Visualization ###
cols = st.columns(num_cell_types)

# Vis reference
for col_i, name_i in zip(cols, cell_type_names):
    seq_i, ctcf_i, atac_i, mat_i = cell_data[name_i]
    col_i.header(name_i)

    import cv2
    log_mat = np.log(mat_i + 1).astype(float)
    resized_mat = cv2.resize(log_mat, (256, 256), interpolation=cv2.INTER_LINEAR)
    mat_i = resized_mat

    vis.plot_mat(col_i, mat_i, cell_type = cell_type_names, loci = loci_info)

# Vis models
for model_name in model_names:
    for col_i, name_i in zip(cols, cell_type_names):
        mat_i = models_preds[name_i][model_name]
        vis.plot_mat(col_i, mat_i, cell_type = cell_type_names, loci = loci_info)
