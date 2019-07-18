from torch.nn.modules.loss import MSELoss
from LSTM_FCN import AutoEncoder
import Learner as Lrn
from pathlib import Path
from util.Subject import Subject, PairedWindows, Window3D  # requied for unpickling
import argparse
from matplotlib import pyplot as plt
import re
import json
import os


def get_experiment_path():
    def get_num(file_name):
        return int(re.search(r'(\d*)$', str(file_name)).group(1))

    try:
        exp_id = max(map(get_num, exp_dir.iterdir())) + 1
    except:
        exp_id = 1
    exp_path = Path(exp_dir / '_'.join(['exp', str(exp_id)]))
    os.mkdir(exp_path)
    return exp_path


class Experiment:
    def __init__(self, train_window, train_ratio, exp_path):
        self.train_losses, self.test_losses = None, None
        self.exp_path = exp_path

        self.md = Lrn.LearnerMetaData(train_windows=train_window,
                                      train_ratio=train_ratio,
                                      loss_lambda=1,
                                      latent_vector_size=10)
        self.dl = Lrn.DataLoaders(Path('Data/3D').iterdir(), self.md)
        self.model = AutoEncoder(self.md)
        self.learner = Lrn.SimpleLearner(self.dl.train_dl,
                                         self.dl.test_dl,
                                         self.model,
                                         loss_func=MSELoss(),
                                         loss_lambda=self.md.loss_lambda)

        with open(str(self.exp_path / 'exp_meta.txt'), 'w') as jf:
            json.dump({**{'train_subjects': [str(s) for s in self.dl.train_subjects],
                          'test_subjects': [str(s) for s in self.dl.test_subjects]},
                       **self.md.to_json()}, jf)

        self.regressor = None

    def run(self, epochs, lr, plot=True):
        self.train_losses, self.test_losses = self.learner.fit(epochs=epochs, lr=lr)

        fig, ax = plt.subplots()
        ax.plot(self.train_losses, label='train_loss')
        ax.plot(self.test_losses, label='test_loss')
        ax.legend()

        plt.savefig(self.exp_path / 'res')
        if plot:
            plt.show()

    def classification_eval(self): self.learner.classification_eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-windows', '-w', default=2)
    parser.add_argument('-ratio', '-r', default=0.7)
    args = parser.parse_args()

    exp_dir = Path('Experiments')
    experiment = Experiment(args.windows, args.ratio, get_experiment_path())
    experiment.run(epochs=100, lr=1e-3)
    experiment.classification_eval()
