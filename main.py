import Subject
from torch.nn.modules.loss import BCELoss
from LSTM_FCN import LSTMFCN
import Learner as Lrn

if __name__ == '__main__':
    subject = Subject.create_subject()
    model = LSTMFCN(subject.min_w, 1)
    loss_func = BCELoss()
    dl = Lrn.DataLoaders(subject, Lrn.LearnerMetaData())
    learner = Lrn.SimpleLearner([dl.train_dl, dl.test_dl], model, loss_func)

    losses = learner.fit()

