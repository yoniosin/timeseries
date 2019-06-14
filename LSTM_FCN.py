import torch
import torch.nn as nn


class BlockLSTM(nn.Module):
    def __init__(self, time_steps, num_variables, lstm_hs=256, dropout=0.8, attention=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=time_steps, hidden_size=lstm_hs, num_layers=num_variables)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # input is of the form (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = torch.transpose(x, 0, 1)
        # lstm layer is of the form (num_variables, batch_size, time_steps)
        x, _ = self.lstm(x)
        # dropout layer input shape:
        y = self.dropout(x)
        # output shape is of the form ()
        return y


class BlockFCNConv(nn.Module):
    def __init__(self, in_channel=1, out_channel=128, kernel_size=8, momentum=0.99, epsilon=0.001, squeeze=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=out_channel, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (batch_size, num_variables, time_steps), e.g. (128, 1, 512)
        x = self.conv(x)
        # input (batch_size, out_channel, L_out)
        x = self.batch_norm(x)
        # same shape as input
        y = self.relu(x)
        return y


class BlockFCNConv2D(nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=1, momentum=0.99, epsilon=0.001):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.batch_norm = nn.BatchNorm1d(num_features=1, eps=epsilon, momentum=momentum)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=248, out_channels=1, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv1(x).squeeze(0)
        x = torch.transpose(x, 0, 1)
        x = self.batch_norm(x)
        x = self.relu(x)
        y = self.conv2(torch.transpose(x, 0, 1))
        return y


class BlockFCN(nn.Module):
    def __init__(self, time_steps, channels=[1, 128, 256, 128], kernels=[8, 5, 3], mom=0.99, eps=0.001):
        super().__init__()
        self.conv1 = BlockFCNConv(channels[0], channels[1], kernels[0], momentum=mom, epsilon=eps, squeeze=True)
        self.conv2 = BlockFCNConv(channels[1], channels[2], kernels[1], momentum=mom, epsilon=eps, squeeze=True)
        self.conv3 = BlockFCNConv(channels[2], channels[3], kernels[2], momentum=mom, epsilon=eps)
        output_size = time_steps - sum(kernels) + len(kernels)
        self.global_pooling = nn.AvgPool1d(kernel_size=output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # apply Global Average Pooling 1D
        y = self.global_pooling(x)
        return y


class LSTMFCN(nn.Module):
    def __init__(self, time_steps, num_variables):
        super().__init__()
        self.lstm_block = BlockLSTM(time_steps, num_variables, lstm_hs=128)
        self.fcn_block = BlockFCN(time_steps)
        self.linear = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        # input is (batch_size, time_steps), it has to be (batch_size, 1, time_steps)
        x = x.unsqueeze(1)
        # pass input through LSTM block
        x1 = self.lstm_block(x)
        x1 = torch.transpose(x1, 1, 2)
        # pass input through FCN block
        x2 = self.fcn_block(x)
        # concatenate blocks output
        x = torch.cat([x1, x2], 1)
        # pass through Softmax activation
        y = self.linear(x.squeeze())

        return y


class AmygNet(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_conv = BlockFCNConv2D(in_channels=in_channel).double()
        self.LstmFcn = LSTMFCN(18, 1).double()

    def forward(self, x):
        x1 = torch.squeeze(self.in_conv(x), dim=1)
        y = self.LstmFcn(x1)

        return y


