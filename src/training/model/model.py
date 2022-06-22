import torch
import torch.nn as nn

import model.blocks as blocks

def main():
    model = ConvModel(2).cuda()
    inputs = torch.rand(4, 2097152, 5).cuda()
    output = model(inputs)
    print(output.shape, output.mean(), output.var())

class ConvModel(nn.Module):
    def __init__(self, num_genomic_features):
        super(ConvModel, self).__init__()
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = 256, num_blocks = 12)
        self.decoder = blocks.Decoder(512)

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

    def move_feature_forward(self, x):
        '''
        input dim:
        bs, img_len, feat
        to: 
        bs, feat, img_len
        '''
        return x.transpose(1, 2)

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map

class ConvTransModel(ConvModel):
    
    def __init__(self, num_genomic_features):
        super(ConvTransModel, self).__init__(num_genomic_features)
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = 256, num_blocks = 12)
        self.attn = blocks.AttnModule(hidden = 256)
        self.decoder = blocks.Decoder(512)
    
    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.move_feature_forward(x)
        x = self.attn(x)
        x = self.move_feature_forward(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

class ConvDilatedModel(ConvModel):
    
    def __init__(self, num_genomic_features):
        super(ConvDilatedModel, self).__init__(num_genomic_features)
        self.encoder = blocks.EncoderSplit(num_genomic_features, output_size = 256, num_blocks = 12)
        self.dila_blocks = blocks.DilatedModule(hidden = 256)
        self.decoder = blocks.Decoder(512)
    
    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        x = self.dila_blocks(x)
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x

if __name__ == '__main__':
    main()
