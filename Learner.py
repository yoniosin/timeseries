import torch
from torch.utils.data import DataLoader, TensorDataset
from util.config import LearnerMetaData
from tqdm import tqdm
from typing import Iterable
from pathlib import Path
import random
import pickle
import numpy as np


def reshape_input(x): return torch.transpose(x.squeeze(), 0, 2)


class SimpleLearner:
    def __init__(self, train_data, test_data, model, loss_func, loss_lambda):
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.loss_func = loss_func
        self.loss_lambda = loss_lambda

    def update(self, x, y, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr)
        loss = self.calc_loss(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    def calc_loss(self, x, y):
        x_rec, y_hat = self.model(x)
        reconstruction_loss = self.loss_func(x, x_rec)
        regression_loss = self.loss_func(y, y_hat)
        return regression_loss + self.loss_lambda * reconstruction_loss

    def fit(self, epochs=1, lr=1e-3):
        self.model.train()
        train_losses = []
        test_loss = []
        for i in tqdm(range(epochs)):
            losses = []
            for x, y in self.train_data:
                x = torch.transpose(x.squeeze(), 0, 2)
                current_loss = self.update(x, y, lr)
                losses.append(current_loss)
            train_losses.append(np.average(losses))
            test_loss.append(self.evaluate())

        torch.save(self.model.state_dict(), 'train.pt')
        return train_losses, test_loss

    def evaluate(self):
        self.model.eval()
        res = []
        for x, y in self.test_data:
            x = reshape_input(x)
            res.append(self.calc_loss(x, y).item())
        self.model.train()
        return np.average(res)

    def classification_eval(self):
        y = []
        y_pred = []
        self.model.eval()
        for x, y_real in self.test_data:
            x = torch.transpose(x.squeeze(), 0, 2)
            _, y_pred_tmp = self.model(x)
            y.append(int(y_real > 0))
            y_pred.append(int(y_pred_tmp > 0))

        print(confusion_matrix(y, y_pred))
        self.model.train()


class DataLoaders:
    def __init__(self, subjects: Iterable[Path], md: LearnerMetaData):
        def split_subjects():
            def add_window_to_list(path):
                l = self.train_subjects if random.random() < md.train_ratio else self.test_subjects
                l.append(path)

            [add_window_to_list(s) for s in subjects]
            self.train_len = len(self.train_subjects)
            self.test_len = len(self.test_subjects)

        def build_data_loader(path_list: Iterable[Path], fold):
            def get_sub_data(path: Path):
                subject = pickle.load(open(str(path), 'rb'))
                return subject.get_data(train_num=md.train_windows, width=md.min_w, scalar_result=False)

            data = [get_sub_data(p) for p in path_list]
            X, y = list(zip(*data))

            self.X = torch.stack(X).double()
            self.y = torch.stack(y).double()

            data_set = torch.stack([torch.cat(subject, dim=3) for subject in data]).double()
            torch.save(data_set, open('_'.join(('3d', fold, 'dataset.pt')), 'wb'))

            return DataLoader(TensorDataset(self.X, self.y), shuffle=False)

        self.train_subjects = []
        self.test_subjects = []
        split_subjects()
        self.train_dl = build_data_loader(self.train_subjects, 'train')
        self.test_dl = build_data_loader(self.test_subjects, 'test')


if __name__ == '__main__':
    md = LearnerMetaData(train_windows=2,
                         train_ratio=0.7,
                         loss_lambda=1,
                         latent_vector_size=10)
    dl = DataLoaders(Path('Data/3D').iterdir(), md)
