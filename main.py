from torch.nn.modules.loss import MSELoss
from LSTM_FCN import AmygNet
import Learner as Lrn
from pathlib import Path
from PreProcess.PreProcess import Subject, PairedWindows, Window, Voxel # requied for unpickling
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_windows', '-tw', default=2)
    parser.add_argument('-train_ratio', '-tr', default=0.7)
    parser.add_argument('-lstm_in', default=8)
    parser.add_argument('-lstm_layers', default=3)
    args = parser.parse_args()

    md = Lrn.LearnerMetaData(train_windows=args.train_windows, train_ratio=args.train_ratio)
    dl = Lrn.DataLoaders(Path('Data').iterdir(), md)
    model = AmygNet(in_channels=md.in_channels,
                    lstm_in_channels=args.lstm_in,
                    lstm_layers=args.lstm_layers,
                    time_steps=md.min_w)
    loss_func = MSELoss()
    learner = Lrn.SimpleLearner([dl.train_dl, dl.test_dl], model, loss_func)

    losses = learner.fit()
    print(losses)
