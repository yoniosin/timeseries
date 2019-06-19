import torch
from torch.utils.data import DataLoader, TensorDataset
from PreProcess import Subject
from dataclasses import dataclass
import random
from tqdm import tqdm
from typing import Set


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
    def __init__(self, subjects: Set[Subject], md: LearnerMetaData):
        train_windows = set(random.sample(subjects, int(md.train_ratio * md.total_subjects)))
        test_windows = subjects.difference(train_windows)

        def build_data_loader(subjects_sub: Set[Subject]):
            data = [s.get_data(md.train_windows) for s in subjects_sub]
            X, y = list(zip(*data))

            X = torch.tensor(X).double()
            y = torch.tensor(y).double()

            return DataLoader(TensorDataset(X, y), shuffle=False)

        self.train_dl = build_data_loader(train_windows)
        self.test_dl = build_data_loader(test_windows)
