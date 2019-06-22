import torch
from torch.utils.data import DataLoader, TensorDataset
from config import LearnerMetaData
from tqdm import tqdm
from typing import Iterable
from pathlib import Path
import random
import pickle


class SimpleLearner:
    def __init__(self, data, model, loss_func):
        self.data, self.model, self.loss_func = data, model, loss_func

    def update(self, x, y, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    def fit(self, epochs=1, lr=1e-3):
        losses = []
        for _ in tqdm(range(epochs)):
            for x, y in self.data[0]:
                current_loss = self.update(x, y, lr)
                losses.append(current_loss)
        return losses


class DataLoaders:
    def __init__(self, subjects: Iterable[Path], md: LearnerMetaData):
        train_windows = []
        test_windows = []

        def add_window_to_list(path):
            l = train_windows if random.random() < md.train_ratio else test_windows
            l.append(path)

        [add_window_to_list(s) for s in subjects]

        def build_data_loader(path_list: Iterable[Path]):
            def get_sub_data(path: Path):
                subject = pickle.load(open(str(path), 'rb'))
                return subject.get_data(train_num=md.train_windows, width=md.min_w)

            data = [get_sub_data(p) for p in path_list]
            X, y = list(zip(*data))

            X = torch.tensor(X).double()
            y = torch.tensor(y).double()

            return DataLoader(TensorDataset(X, y), shuffle=False)

        self.train_len = len(train_windows)
        self.test_len = len(test_windows)
        self.train_dl = build_data_loader(train_windows)
        self.test_dl = build_data_loader(test_windows)
