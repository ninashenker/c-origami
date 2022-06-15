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

from model_class import ModelTemplate
from pineapple.model import CNN_dual_encoder


class CustomModel(ModelTemplate):

  def load_model(self, model_save):
    self.model = CNN_dual_encoder(7)
    self.model = torch.nn.DataParallel(self.model)
    self.model.load_state_dict(torch.load(model_save, map_location=self.device))
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

    return seq, ctcf, atac

@hydra.main(version_base=None, config_path=".", config_name="config")
def inference(cfg):
  print(OmegaConf.to_yaml(cfg))
  print(cfg.inference.task)

  #Model Selection
  model = get_model(cfg.inference.model_path)

  #Ensure start position is before chromosome length.
  start_pos = cfg.inference.start_pos
  lengths_by_chr = locus_config(cfg.inference.chr_lengths)
  chr_length = lengths_by_chr[str(cfg.inference.chr_num)]
  assert start_pos < chr_length, f"{start_pos} must be less than {chr_length}"

  #Get data.
  chr_name = f"chr{cfg.inference.chr_num}"
  seq, ctcf, atac = get_data(
      chr_name,
      start_pos,
      cfg.inference.chr_fa_path,
      os.path.join(cfg.inference.input_folder, cfg.inference.ctcf_name),
      os.path.join(cfg.inference.input_folder, cfg.inference.atac_name),
  )

  pred = model.predict(seq, ctcf, atac)

if __name__ == "__main__":
  inference()
