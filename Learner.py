import torch
from torch.utils.data import DataLoader, TensorDataset
from config import LearnerMetaData
from tqdm import tqdm
from typing import Iterable
from pathlib import Path
import random
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix


class SimpleLearner:
    def __init__(self, train_data, test_data, model, loss_func):
        self.train_data = train_data
        self.test_data = test_data
        self.model = model
        self.loss_func = loss_func

    def update(self, x, y, lr):
        opt = torch.optim.Adam(self.model.parameters(), lr)
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        return loss.item()

    def fit(self, epochs=1, lr=1e-3):
        self.model.train()
        train_losses = []
        test_loss = []
        for i in tqdm(range(epochs)):
            losses = []
            for x, y in self.train_data:
                current_loss = self.update(x, y, lr)
                losses.append(current_loss)
            train_losses.append(np.average(losses))
            test_loss.append(self.evaluate())

        torch.save(self.model.state_dict(), 'train.pt')
        return train_losses, test_loss

    def evaluate(self):
        self.model.eval()
        res = np.average([self.loss_func(self.model(x), y).item() for x, y in self.test_data])
        self.model.train()
        return res

    def classification_eval(self):
        y = []
        y_pred = []
        self.model.eval()
        for x, y_real in self.test_data:
            y.append(int(y_real > 0))
            y_pred.append(int(self.model(x) > 0))

        print(confusion_matrix(y, y_pred))
        self.model.train()


class DataLoaders:
    def __init__(self, subjects: Iterable[Path], md: LearnerMetaData):
        self.train_subjects = []
        self.test_subjects = []

        def add_window_to_list(path):
            l = self.train_subjects if random.random() < md.train_ratio else self.test_subjects
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

        self.train_len = len(self.train_subjects)
        self.test_len = len(self.test_subjects)
        self.train_dl = build_data_loader(self.train_subjects)
        self.test_dl = build_data_loader(self.test_subjects)
