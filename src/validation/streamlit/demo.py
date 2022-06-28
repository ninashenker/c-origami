import os
import sys
sys.path.append('../')
sys.path.append('../../')
import streamlit as st
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import font_manager
font = '/gpfs/home/jt3545/fonts/Arial.ttf'
font_manager.fontManager.addfont(font)
import plot_config
config = plot_config.config
plt.rcParams.update(config)
mm = 1/25.4

import vis
import st_utils

data_loc = 'bpp' #'local'

### Model Selection ###
model_st, model_cell_st, epoch_st = st_utils.model_config(st, ['pineapple', 'tomato', 'eggplant', 'peach', 'pear'])
#model_st, model_cell_st, epoch_st = st_utils.model_config(st)

# Refresh model
import importlib
sys.path.append('../../')
from models.model_class import CellType 
model_kit = importlib.import_module(f'models.{model_st}.model_kit')

@st.cache(allow_output_mutation=True)
def get_model(model_st, model_cell_st, epoch_st):
    model_dir = st_utils.get_model_path(data_loc, model_st, model_cell_st, epoch_st)
    model = model_kit.CustomModel(model_dir)
    print(model.model_id())
    return model

#os.chdir('/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-visualization/src/streamlit')

model = get_model(model_st, model_cell_st, epoch_st)


### Locus Selection ###
chr_num, start = st_utils.locus_config(st, chr_idx = 14, start_init = 59100000)

### Load Cell ###
@st.cache(allow_output_mutation=True, hash_funcs={'cell_imr90.ctcf.keys()': id})
def get_cells(ctcf_name, atac_name):
    cell_imr90 = CellType(st_utils.get_cell_path(data_loc, 'imr90'), ctcf_name, atac_name)
    cell_gm12878 = CellType(st_utils.get_cell_path(data_loc, 'gm12878'), ctcf_name, atac_name)
    cell_h1 = CellType(st_utils.get_cell_path(data_loc, 'h1'), ctcf_name, atac_name)
    cell_k562 = CellType(st_utils.get_cell_path(data_loc, 'k562'), ctcf_name, atac_name)
    cell_tcell = CellType(st_utils.get_cell_path(data_loc, 'tcell'), ctcf_name, atac_name)
    return cell_imr90, cell_gm12878, cell_h1, cell_k562, cell_tcell

ctcf_name, atac_name = st_utils.get_ctcf_atac_names(model_st)
cell_imr90, cell_gm12878, cell_h1, cell_k562, cell_tcell = get_cells(ctcf_name, atac_name)

### Plotting ###
input_data = {}
pred = {}
truth = {}

input_data['IMR-90'] = cell_imr90.get_data(chr_num, start)
input_data['GM'] = cell_gm12878.get_data(chr_num, start)
input_data['tcell'] = cell_tcell.get_data(chr_num, start)

cell_types = ['IMR-90', 'tcell']

for cell_type in cell_types:
    pred[cell_type], truth[cell_type] = model.predict(*input_data[cell_type])

# GM12878 Correction
#truth['GM'] += 0.1
truth['tcell'] += 1.1

loci_info = [chr_num, str(start)]

def get_insulation(matrix):

    from insulation_score import chr_score

    truth_scores = np.array(chr_score(matrix))
    truth_scores = (truth_scores - np.nanmean(truth_scores)) / np.nanstd(truth_scores)
    return truth_scores

def compare_pair(cell_type1, cell_type2):
    col1.header(cell_type1)
    col2.header(cell_type2)
    col3.header(f'{cell_type2} - {cell_type1}')

    vis.plot_mat(col1, truth[cell_type1], cell_type = cell_type1, loci = loci_info)
    vis.plot_insulation(col1, truth[cell_type1], cell_type = cell_type1, loci = loci_info)
    insu_truth_1 = get_insulation(truth[cell_type1]) # Get insulation for diff calculation, same below
    vis.plot_mat(col2, truth[cell_type2], cell_type = cell_type2, loci = loci_info)
    vis.plot_insulation(col2, truth[cell_type2], cell_type = cell_type2, loci = loci_info)
    insu_truth_2 = get_insulation(truth[cell_type2])
    vis.plot_diff(col3, truth[cell_type2] - truth[cell_type1], img_type = 'diff_truth', cell_type = f'{cell_type2}_{cell_type1}', loci = loci_info)
    vis.vis_insulation(col3, insu_truth_2 - insu_truth_1, img_type = 'truth_diff', cell_type = f'{cell_type2}_{cell_type1}', loci = loci_info)

    vis.plot_mat(col1, pred[cell_type1], img_type = 'pred', cell_type = cell_type1, loci = loci_info)
    vis.plot_insulation(col1, pred[cell_type1], img_type = 'pred', cell_type = cell_type1, loci = loci_info)
    insu_pred_1 = get_insulation(pred[cell_type1]) 
    vis.plot_mat(col2, pred[cell_type2], img_type = 'pred', cell_type = cell_type2, loci = loci_info)
    vis.plot_insulation(col2, pred[cell_type2], img_type = 'pred', cell_type = cell_type2, loci = loci_info)
    insu_pred_2 = get_insulation(pred[cell_type2])
    vis.plot_diff(col3, pred[cell_type2] - pred[cell_type1], img_type = 'diff_pred', cell_type = f'{cell_type2}_{cell_type1}', loci = loci_info)
    vis.vis_insulation(col3, insu_pred_2 - insu_pred_1, img_type = 'pred_diff', cell_type = f'{cell_type2}_{cell_type1}', loci = loci_info)

    ctcf1 = vis.ctcf_proc(input_data[cell_type1][1], model_st)
    atac1 = vis.atac_proc(input_data[cell_type1][2], model_st)
    ctcf2 = vis.ctcf_proc(input_data[cell_type2][1], model_st)
    atac2 = vis.atac_proc(input_data[cell_type2][2], model_st)

    vis.plot_epi(col1, ctcf1, atac1, cell_type = cell_type1, loci = loci_info)
    vis.plot_insulation_diff(col1, truth[cell_type1], pred[cell_type1], 'truth', 'pred', img_type = 'diff', cell_type = f'{cell_type1}', loci = loci_info)
    vis.plot_epi(col2, ctcf2, atac2, cell_type = cell_type2, loci = loci_info)
    vis.plot_insulation_diff(col2, truth[cell_type2], pred[cell_type2], 'Target', 'Prediction', img_type = 'diff', cell_type = f'{cell_type2}', loci = loci_info)
    vis.plot_epi_diff(col3, ctcf1, ctcf2, atac1, atac2, cell_type = f'{cell_type2}_{cell_type1}', loci = loci_info)
    vis.vis_insulation_diff(col3, insu_truth_2 - insu_truth_1, insu_pred_2 - insu_pred_1, 'Target', 'Prediction', img_type = 'diff', cell_type = f'{cell_type2}_{cell_type1}', loci = loci_info)

col1, col2, col3 = st.columns(3)

compare_pair('IMR-90', 'tcell')
