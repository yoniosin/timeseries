import Subject
from torch.nn.modules.loss import MSELoss
from LSTM_FCN import AmygNet
import Learner as Lrn
from copy import copy

if __name__ == '__main__':
    subject = Subject.create_subject()
    num_subjects = 5
    model = AmygNet(num_subjects)
    loss_func = MSELoss()
    dl = Lrn.DataLoaders(set([copy(subject) for _ in range(num_subjects)]), Lrn.LearnerMetaData(5))
    learner = Lrn.SimpleLearner([dl.train_dl, dl.test_dl], model, loss_func)

    losses = learner.fit()

