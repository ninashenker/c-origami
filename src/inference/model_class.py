import torch
import numpy as np
import pandas as pd
import pyBigWig as pbw

class ModelTemplate():
    
    def __init__(self, model_save):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Using: ', self.device)
        self.image_scale = 256 # IMPORTANT, scale 210 to 256
        self.model = None
        self.load_model(model_save)

    def load_model(self, model_save):
        raise Exception('load model not implemented')

    def process_data(self, seq, ctcf, atac, mat):
        raise Exception('process data not implemented')

    def predict(self, seq, ctcf, atac, mat):
        raise Exception('process data not implemented')

    def encode_seq(self, seq):
        ''' 
        encode dna to onehot (n x 5)
        '''
        seq_emb = np.zeros((len(seq), 5))
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb

