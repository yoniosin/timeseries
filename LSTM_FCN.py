import torch.nn as nn
from util.config import LearnerMetaData


class Encoder(nn.Module):
    def __init__(self, md: LearnerMetaData):
        super().__init__()
        self.linear_windows = nn.Linear(md.in_channels, 1).double()
        self.dropout = nn.Dropout()
        self.linear_voxels = nn.Linear(md.voxels_num, md.latent_vector_size).double()
        self.relu = nn.ReLU()
        self.linear_time = nn.Linear(md.min_w, 1).double()

    def forward(self, x):
        x1 = self.dropout(self.linear_windows(x))
        x2 = self.relu(self.linear_voxels(x1.squeeze()))
        z = self.linear_time(x2.transpose(0, 1))

        return z


class Decoder(nn.Module):
    def __init__(self, md: LearnerMetaData):
        super().__init__()

        self.linear_time = nn.Linear(1, md.min_w).double()
        self.tan = nn.Tanh()
        self.linear_voxels = nn.Linear(md.latent_vector_size, md.voxels_num).double()
        self.tan2 = nn.Tanh()
        self.linear_windows = nn.Linear(1, md.in_channels).double()

    def forward(self, x):
        x1 = self.tan(self.linear_time(x.unsqueeze(0)))
        x2 = self.tan2(self.linear_voxels(x1.transpose(2, 1)))
        y = self.linear_windows(x2.transpose(0, 2))

        return y


class AutoEncoder(nn.Module):
    def __init__(self, md: LearnerMetaData):
        super().__init__()
        self.encoder = Encoder(md)
        self.decoder = Decoder(md)
        self.linear = nn.Linear(md.latent_vector_size, 1).double()

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        y = self.linear(z.transpose(0, 1))

        return x_rec.transpose(0, 1), y.squeeze(0)

