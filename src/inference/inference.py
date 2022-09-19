from omegaconf import DictConfig, OmegaConf
import hydra
import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import importlib
import json
import pyBigWig as pbw
from matplotlib.colors import LinearSegmentedColormap

sys.path.append("../training")
from main import TrainModule

from model_class import ModelTemplate
from pineapple.model import CNN_dual_encoder
import pytorch_lightning as pl


class CustomModel(ModelTemplate):

  def load_model(self, model_save):
    #self.model = CNN_dual_encoder(7)
    #self.model = torch.nn.DataParallel(self.model)
    #self.model.load_state_dict(torch.load(model_save, map_location=self.device))
    self.model = TrainModule.load_from_checkpoint(model_save)
    self.model.eval()

  def process_data(self, seq, ctcf, atac):
      seq_emb = torch.tensor(self.encode_seq(seq))
      # CTCF processing
      ctcf = np.nan_to_num(ctcf, 0) # Important! replace nan with 0
      #log_ctcf = np.log(ctcf + 1)
      log_ctcf = ctcf

      log_ctcf = torch.tensor(log_ctcf)
      # ATAC seq processing
      atac = np.nan_to_num(atac, 0)
      log_atac = torch.tensor(np.log(atac + 1))

      return seq_emb, log_ctcf, log_atac

  def predict(self, seq, ctcf, atac):
    with torch.no_grad():
      seq, ctcf, atac = self.process_data(seq, ctcf, atac)
      seq = seq.to(self.device).unsqueeze(0)
      ctcf = ctcf.to(self.device).unsqueeze(0)
      atac = atac.to(self.device).unsqueeze(0)

      inputs = torch.cat([seq, ctcf.unsqueeze(2), atac.unsqueeze(2)], dim = 2)
      outputs = self.model(inputs).squeeze(0)
      return outputs.detach().cpu().numpy()

  def perturbation(self, seq, ctcf, atac, del_pos, window, start_pos):
    for pos in del_pos:
        start = pos['start'] - start_pos
        end = pos['end'] - start_pos
        seq[start:end] = 4
        ctcf[start:end] = 0
        atac[start:end] = 0
    len_diff = window - len(seq)
    if len_diff < 0:
        raise Exception('Deletion range should be smaller than sequence length (2 mb)')
    return seq, ctcf, atac

  def screening(self, seq, ctcf, atac, start_pos, end_pos, del_window, step_size, window, chr_num):
    #scores, chr_nums, start_poss, end_poss = ([] for i in range(4))
    with open(screening.bedgraph, 'w') as bg:
      for window_start in range(start_pos, end_pos, step_size):
        window_end = window_start + del_window
        del_pos = [{'start': window_start, 'end': window_end}]
        screened_seq, screened_ctcf, screened_atac = self.perturbation(seq.copy(), ctcf.copy(), atac.copy(), del_pos, window, start_pos)
        print('screened')

        # Calculate impact score on screened sequences
        pred = self.predict(seq, ctcf, atac)
        screened_pred = self.predict(screened_seq, screened_ctcf, screened_atac)
        score = np.abs(pred - screened_pred).mean()

        # Add impact score to bedgraph file
        line  = f"{chr_num}\t{window_start}\t{window_end}\t{score}\n"
        bg.write(f'{line}')
    return bg

def locus_config(lengths_dir):
  with open(lengths_dir, 'r') as len_file:
    chr_lengths = json.load(len_file)

  return chr_lengths

def get_model(model_path):
    model = CustomModel(model_path)
    return model

def read_seq(dna_path):
    '''
    Transform fasta data to numpy array

    Args:
        dna_path (str): Directory to DNA .fa path

    Returns:
        array: A numpy char array that contains DNA for a chromosome
    '''
    with open(dna_path, 'r') as f:
        seq = f.read()
    seq = seq[seq.find('\n'):]
    seq = seq.replace('\n', '').lower()
    return seq


def seq_to_npy(seq, start, end):
    '''
    Transform fasta data to numpy array

    Args:
        dna_dir (str): Directory to DNA .fa path

    Returns:
        array: A numpy char array that contains DNA for a chromosome
    '''
    seq = seq[start : end]
    en_dict = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
    en_seq = [en_dict[ch] for ch in seq]
    np_seq = np.array(en_seq, dtype = int)
    return np_seq


def bw_to_np(bw_file, chr_name, start, end):
    signals = bw_file.values(chr_name, start, end)
    return np.array(signals)


def get_data(chr_num, start, chr_fa_path, ctcf_path, atac_path, window = 2097152):
    end = start + window

    print('Loading fa')
    seq_chr_name = read_seq(f'{chr_fa_path}/{chr_num}.fa')
    seq = seq_to_npy(seq_chr_name, start, end)

    print('Loading ctcf')
    with pbw.open(ctcf_path) as ctcf_f:
      ctcf = bw_to_np(ctcf_f, chr_num, start, end)

    print('Loading atac')
    with pbw.open(atac_path) as atac_f:
      atac = bw_to_np(atac_f, chr_num, start, end)

    return seq, ctcf, atac, window

#Visualize prediction
def plot_mat(name, image, chr_num, chr_start, task, del_pos=None):
    color_map = LinearSegmentedColormap.from_list("bright_red",
                                                [(1,1,1),(1,0,0)])
    fig, ax = plt.subplots()
    ax.imshow(image, cmap = color_map, vmin = 0, vmax = 5)
    ax.set_axis_off()
    #hic_chart = handle.pyplot(fig)
    if not os.path.isdir("prediction_matrix_output"):
      os.makedirs("prediction_matrix_output")

    name = f"{task}_{name}_{chr_num}_{chr_start}"
    name = name + f"_{del_pos}" if del_pos is not None else name
    plt.savefig(f'prediction_matrix_output/{name}.png')

@hydra.main(version_base=None, config_path=".", config_name="config")
def inference(cfg):
  print(OmegaConf.to_yaml(cfg))
  print(cfg.del_pos)
  #Model Selection
  model = get_model(cfg.model_path)

  #Ensure start position is before chromosome length.
  start_pos = cfg.start_pos
  lengths_by_chr = locus_config(os.path.join(cfg.chr_fa_path, cfg.chr_lengths))
  chr_length = lengths_by_chr[str(cfg.chr_num)]
  assert start_pos < chr_length, f"{start_pos} must be less than {chr_length}"

  #Get data.
  chr_name = f"chr{cfg.chr_num}"
  seq, ctcf, atac, window = get_data(
      chr_name,
      start_pos,
      cfg.chr_fa_path,
      os.path.join(cfg.input_folder, cfg.ctcf_name),
      os.path.join(cfg.input_folder, cfg.atac_name),
  )

  del_pos = None
  task = cfg.task
  if task == 'perturbation':
    print('Perturbation task')
    del_pos = cfg.del_pos
    seq, ctcf, atac  = model.perturbation(seq, ctcf, atac, del_pos, window, start_pos)
  
  elif task == 'screening':
    print('Screening task')
    end_pos = cfg.end_pos
    del_window = cfg.del_window
    step_size = cfg.step_size
    chr_screening = model.screening(seq, ctcf, atac, start_pos, end_pos, del_window, step_size, window, chr_name)
    
  pred = model.predict(seq, ctcf, atac)
  cell_header = cfg.cell_line
  plot_mat(cell_header, pred, chr_name, start_pos, task, del_pos)


if __name__ == "__main__":
  inference()

