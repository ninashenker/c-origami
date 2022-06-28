import re
import json
import copy
import numpy as np

# Model Section
def model_config(st, code_set = None):
    model_dict = {'eggplant' : {},
                  'grape' : {},
                  'durian' : {},
                  'peach' : {},
                  'tomato' : {},
                  'mango' : {},
                  'mango_seq' : {},
                  'pear' : {},
                  'pineapple' : {}}
    # All epoches and default epoch
    model_dict['eggplant'] = {'IMR90' : ([57], 0),
                              'GM12878' : ([46], 0)}
    model_dict['grape'] = {'IMR90' : ([14, 48, 71, 88], 3)}
    model_dict['durian'] = {'IMR90' : ([41, 57, 61, 123, 136], 1)}
    model_dict['peach'] = {'IMR90' : ([21, 43, 54, 63, 87, 98, 119], 1)}
    model_dict['tomato'] = {'IMR90' : ([60, 82], 0),
                            'CUTLL1' : ([27, 56, 78, 90], 0)}
    model_dict['mango'] = {'IMR90' : ([50, 66, 75], 1)}
    model_dict['mango_seq'] = {'IMR90' : ([28, 48], 1)}
    model_dict['pear'] = {'IMR90' : ([54, 62], 0)}
    model_dict['pineapple'] = {'IMR90' : ([49, 69], 0)}

    # Filter by code set
    if not code_set is None:
        model_dict = {item : model_dict[item] for item in code_set}

    model_list = list(model_dict.keys())
    st.sidebar.title('Model')
    model_st = st.sidebar.selectbox('Model Code', model_list, index = 0)
    model_cells = list(model_dict[model_st].keys())
    model_cell_st = st.sidebar.selectbox('Training Cell Type', model_cells, index = 0)
    epoch_list = model_dict[model_st][model_cell_st]
    epoch_st = st.sidebar.selectbox('Model Epoch', epoch_list[0], 
                                    index = epoch_list[1])
    return model_st, model_cell_st, epoch_st

def get_model_folder(name):
    path_dict = {'eggplant-imr90' : 'IMR-90_AUGs_2021-10-10T10:20:09.421537',
                 'eggplant-gm12878' : 'run_ADAM_gm12878_2021-10-14T19:16:05.592964',
                 'grape-imr90' : 'IMR-90_AUGs_ctcfIP_2021-10-18T10:07:31.781358',
                 'durian-imr90' : 'IMR-90_peak_2021-11-02T09:09:50.671761',
                 'peach-imr90' : 'IMR-90_peaks_dual_encoder_2021-12-15T16:35:22.569400',
                 'tomato-cutll1' : 'cutll1_bigwig_dual_encoder_chr7_chr9_excluded_2022-01-17T12:55:13.344170',
                 'tomato-imr90' : 'IMR-90_bigwig_dual_encoder_2021-12-30T12:36:35.756762',
                 'mango-imr90' : 'IMR-90_peaks_dual_encoder_dna_only_first_2022-02-02T16:44:49.726230',
                 'mango_seq-imr90' : 'IMR-90_peaks_dual_encoder_dna_only_first_2022-02-02T16:44:49.726230',
                 'pear-imr90' : 'IMR-90_peaks_dual_encoder_dna_ctcf_2022-02-21T18:20:59.545480',
                 'pineapple-imr90' : 'IMR-90_dense_dual_encoder_one_direction_2022-02-27T12:02:28.576473'}
    return path_dict[name]

def get_ctcf_atac_names(model_st):
    # Check model and load corresponding data format
    if model_st in ['eggplant', 'tomato', 'pear', 'pineapple', 'strawberry']:
        ctcf_name = 'final.bigwig'
        atac_name = 'sub_merged.bin1.rpkm.bw'
    elif model_st == 'grape':
        ctcf_name = 'sub_ip.bin1.rpkm.bw'
        atac_name = 'sub_merged.bin1.rpkm.bw'
    elif model_st in ['durian', 'peach', 'mango', 'mango_seq']:
        ctcf_name = 'ctcf_peaks.bigwig'
        atac_name = 'atac_peaks.bigwig'
    return ctcf_name, atac_name

def get_model_path(model_st, model_cell_st, epoch_st, location = 'bpp'):
    model_folder = get_model_folder(f'{model_st}-{model_cell_st.lower()}')
    if location == 'local':
        model_dir = f'/home/jimin/Documents/ingenious-visualization/src/models/{model_st}/models/{model_cell_st.lower()}/state_dict_{epoch_st}.pt'
    elif location == 'bpp':
        model_dir = f'/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-model/model/runs/{model_folder}/state_dict_{epoch_st}.pt'
    else:
        location = None
    return model_dir

def get_cell_path(cell, location = 'bpp'):
    if location == 'local':
        path = f'/home/jimin/Documents/ingenious-test/{cell}'
    elif location == 'bpp':
        path = f'/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-data/{cell}'
    return path

# Locus Section
def get_chr_names(species):
    if species == 'human':
        chrs = list(range(1, 23))
    elif species == 'mouse':
        chrs = list(range(1, 20))
    chrs.append('X')
    chr_names = []
    for chr_num in chrs:
        chr_names.append(f'chr{chr_num}')
    return chr_names

def locus_config(st, species = 'human', data_loc = 'bpp', chr_idx = 1, start_init = 400000):
    st.sidebar.title('Locus')
    # Chromosome
    chr_names = get_chr_names(species)
    chr_num_st = st.sidebar.selectbox('Chromosome', chr_names, index = chr_idx)

    if species == 'human':
        lengths_dir = '/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-evaluation/data/imr90/raw_data/downloads/chr_length.json'
    elif species == 'mouse' and data_loc == 'local':
        lengths_dir = '/home/jimin/Documents/ingenious-test/mesc/chrs/chr_length.json'
    elif species == 'mouse' and data_loc == 'bpp':
        print('loading mouse from bpp')
        lengths_dir = '/gpfs/data/tsirigoslab/home/jt3545/hic_prediction/ingenious-evaluation/data/mesc/raw_data/downloads/chr_length.json'
    with open(lengths_dir, 'r') as len_file:
        chr_lengths = json.load(len_file)

    slider_max = chr_lengths[chr_num_st.split('chr')[1]] - 3000000
    start_st = st.sidebar.slider('Start Position', 0, slider_max, start_init, step = 100000)
    start_input_st = st.sidebar.number_input('Start Position', 0, slider_max, start_st, 100000)
    return chr_num_st, start_input_st

def delete_config(st, total_window = 2097152, start = 0, length = 16384):
    st.sidebar.title('Deletion')
    start_st = st.sidebar.number_input('Start', 0, total_window, start, step = 100000)
    length_st = st.sidebar.number_input('Length', 0, total_window - start_st, length, 16384)
    return start_st, start_st + length_st

def delete(start, end, window, seq, ctcf, atac, mat):
    seq = np.delete(seq, np.s_[start:end], axis = 0)
    ctcf = np.delete(ctcf, np.s_[start:end])
    atac = np.delete(atac, np.s_[start:end])
    return seq[:window], ctcf[:window], atac[:window], mat

def delete_fill(start, end, window, seq, ctcf, atac, mat):
    seq = np.delete(seq, np.s_[start:end], axis = 0)
    ctcf = np.delete(ctcf, np.s_[start:end])
    atac = np.delete(atac, np.s_[start:end])
    len_diff = window - len(seq)
    if len_diff < 0:
        raise Exception('Filled array size should be smaller than target')
    seq = np.pad(seq, (0, len_diff), constant_values = 4)
    ctcf = np.pad(ctcf, (0, len_diff), constant_values = 0)
    atac = np.pad(atac, (0, len_diff), constant_values = 0)
    return seq[:window], ctcf[:window], atac[:window], mat

def delete_front(start, end, window, seq, ctcf, atac, mat):
    start += window
    end += window
    seq = np.delete(seq, np.s_[start:end], axis = 0)
    ctcf = np.delete(ctcf, np.s_[start:end])
    atac = np.delete(atac, np.s_[start:end])
    return seq[-window:], ctcf[-window:], atac[-window:], mat

def shuf(start, end, window, seq, ctcf, atac, mat):

    #print(start)
    #print(end)
    seq_mod = np.zeros_like(seq)
    np.copyto(seq_mod, seq)

    ctcf_mask = ctcf > 0

    shuffled_region = np.random.choice(range(4), end - start)
    seq_mod[start : end] = shuffled_region # Randomize region
    seq_mod[ctcf_mask] = seq[ctcf_mask] # Paste back ctcf protected region

    #print(seq[start : end].sum() - seq_copy[start : end].sum())

    #import pdb; pdb.set_trace()

    return seq_mod, ctcf, atac, mat

def proc_del_images(start, end, raw, pred):
    margin = int((end - start) / 8192)
    start = round(start / 8192)
    end = start + margin 

    canvas = 256
    old_end = canvas - margin
    new_raw = np.zeros_like(raw)
    np.copyto(new_raw, raw)
    #new_raw = raw.copy()
    new_raw[:, start:end] = 0
    new_raw[start:end, :] = 0
    new_pred = np.zeros_like(pred)
    new_pred[:start, :start] = pred[:start, :start]
    new_pred[end:, :start] = pred[start:old_end, :start]
    new_pred[:start, end:] = pred[:start, start:old_end]
    new_pred[end:, end:] = pred[start:old_end, start:old_end]
    print(end - start)
    return new_raw, new_pred

def proc_del_images_front(start, end, raw, pred):
    margin = int((end - start) / 8192)
    start = round(start / 8192)
    end = start + margin 

    new_start = start + margin
    new_raw = np.zeros_like(raw)
    np.copyto(new_raw, raw)
    new_raw[:, start:end] = 0
    new_raw[start:end, :] = 0
    new_pred = np.zeros_like(pred)
    new_pred[:start, :start] = pred[margin : end, margin : end]
    new_pred[end:, :start] = pred[end:, margin : end]
    new_pred[:start, end:] = pred[margin: end, end:]
    new_pred[end:, end:] = pred[end:, end:]
    return new_raw, new_pred

def proc_del_epi(start, end, epi):
    new_epi = np.zeros_like(epi)
    new_epi[:start] = epi[:start]
    new_epi[end:] = epi[start:2097152 - (end - start)]
    return new_epi

def proc_del_epi_front(start, end, epi, window = 2097152):
    start += window
    end += window
    new_epi = np.zeros_like(epi)
    new_epi[:start] = epi[:start]
    new_epi[end:] = epi[start:window - (end - start)]
    return new_epi

def get_sequence(cell_line, chr_num, start, window):
    return cell_line.seq[chr_num][start : start + window]

def get_ctcf_track(sequence):
    #seq_ctcf = 'ccgcg.gg.ggcag'
    #seq_ctcf = 'ggcgc.cc.ccgtc'
    seq_ctcf_forward = 'ccgcg.gg'
    seq_ctcf_reverse = 'ggcgc.cc'

    count = 0
    ctcf_sites = []
    ctcf_track = np.zeros(2097152)
    for match in re.finditer(seq_ctcf_forward, sequence):
        count += 1
        ctcf_sites.append(match.start())
        ctcf_track[match.start() : match.end()] = 5
    print(count)

    for match in re.finditer(seq_ctcf_reverse, sequence):
        print(match.start())
        count += 1
        ctcf_sites.append(match.start())
        ctcf_track[match.start() : match.end()] = 5
    print(count)
    return ctcf_track
