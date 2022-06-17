import torch
import torch.nn as nn

import model.blocks as blocks

def main():
    model = CNN(5).cuda()
    inputs = torch.rand(4, 2097152, 5).cuda()
    output = model(inputs)
    print(output.shape, output.mean(), output.var())

class CNN(nn.Module):
    def __init__(self, input_size):
        super(CNN, self).__init__()
        self.encoder = blocks.Encoder(input_size, output_size = 256, num_blocks = 12)
        # TODO Added
        #self.attention = blocks.AttnModule(hidden = 256, layers = 4)
        '''
        self.header = nn.Sequential(
                            nn.Conv2d(257, 64, 1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 64, 1),
                            nn.BatchNorm2d(64),
                            nn.ReLU(),
                            nn.Conv2d(64, 1, 1),
                            )
        '''
        # TODO added end
        self.decoder = blocks.Decoder(513)
        self.dist_mat = self.gen_distance_matrix(256) # 256 x 256 hic input

    def forward(self, x):
        '''
        Input feature:
        batch_size, length * res, feature_dim
        '''
        x = self.move_feature_forward(x).float()
        x = self.encoder(x)
        # TODO Added
        x = self.move_feature_forward(x)
        #x = self.attention(x)
        x = self.move_feature_forward(x)
        #x = self.outer_cat(x)
        #x = self.append_dist_mat(x)
        #x = self.header(x)
        #x = x.squeeze(1)
        x = self.seq_to_map(x)
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

    def gen_distance_matrix(self, length):
        ''' distance matrix as part of decoder input feature '''
        dis_mat = torch.zeros(1, 1, length, length)
        for i in range(length):
            for j in range(length):
                dis_mat[:, :, i, j] = abs(i - j)
        return dis_mat

    def seq_to_map(self, x):
        x = self.diagonalize(x)
        x = self.append_dist_mat(x)
        return x

    # TODO added
    def outer_cat(self, x):
        x1 = x.unsqueeze(3).repeat(1, 1, 1, 256)
        x2 = x.unsqueeze(2).repeat(1, 1, 256, 1)
        outer = torch.cat([x1, x2], dim = 1)
        return outer

    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map

    def append_dist_mat(self, x):
        batch_size = x.shape[0]
        dist_mat = self.dist_mat.to(x.device).repeat(batch_size, 1, 1, 1)
        x = torch.cat([x, dist_mat], dim = 1)
        return x

class CNN_dual_encoder(CNN):
    def __init__(self, input_size):
        super().__init__(input_size)
        self.encoder = blocks.EncoderSplit(input_size, output_size = 256, num_blocks = 12)

    

if __name__ == '__main__':
    main()
