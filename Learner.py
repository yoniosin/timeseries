import torch
from torch.utils.data import DataLoader, TensorDataset
from Subject import Subject
from dataclasses import dataclass
import random
from tqdm import tqdm
import numpy as np


@dataclass
class LearnerMetaData:
    train_ratio: float = 0.7
    total_windows: int = 5


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
    def __init__(self, subject: Subject, md: LearnerMetaData):
        all_windows = set(range(md.total_windows))
        train_windows_idx = set(random.sample(all_windows, int(md.train_ratio * md.total_windows)))
        test_windows_idx = all_windows.difference(train_windows_idx)

        def build_data_loader(indices):
            def get_windows_avg_diff(idx): return subject.paired_windows[idx].avg_diff()

            def get_window_score(idx): return float(subject.paired_windows[idx].score > 0)

            X = np.array([get_windows_avg_diff(idx) for idx in indices])
            X = torch.tensor(X)
            y = np.array([get_window_score(idx) for idx in indices])
            y = torch.tensor(y)

            return DataLoader(TensorDataset(X, y), shuffle=False)

        self.train_dl = build_data_loader(train_windows_idx)
        self.test_dl = build_data_loader(test_windows_idx)
