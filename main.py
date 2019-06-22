from torch.nn.modules.loss import MSELoss
from LSTM_FCN import AmygNet
import Learner as Lrn
from pathlib import Path
from PreProcess.PreProcess import Subject, PairedWindows, Window, Voxel # requied for unpickling
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-windows', '-w', default=2)
    parser.add_argument('-ratio', '-r', default=0.7)
    args = parser.parse_args()

    md = Lrn.LearnerMetaData(train_windows=args.windows, train_ratio=args.ratio)
    dl = Lrn.DataLoaders(Path('Data').iterdir(), md)
    model = AmygNet(md)
    loss_func = MSELoss()
    learner = Lrn.SimpleLearner([dl.train_dl, dl.test_dl], model, loss_func)

    losses = learner.fit()
    print(losses)
