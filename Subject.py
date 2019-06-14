import scipy.io as sio
import numpy as np
import pickle
from config import SubjectMetaData
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from itertools import chain


class Subject:
    roi = np.where(sio.loadmat('raw_data/roi.mat')['ans'])
    amyg_vox = list(zip(*roi))

    def __init__(self, meta_data: SubjectMetaData):
        def gen_windows(window_type):
            times_list = meta_data.watch_times if window_type == 'watch' else meta_data.regulate_times
            return map(lambda w: Window(*w, window_type, self.bold), enumerate(times_list))

        self.meta_data = meta_data
        self.bold = sio.loadmat(meta_data.bold_mat_path)[meta_data.bold_mat_name]

        self.paired_windows = list(map(PairedWindows, gen_windows('watch'), gen_windows('regulate')))
        self.min_w = min((pw.min_w for pw in self.paired_windows))
        for pw in self.paired_windows:
            pw.min_w = self.min_w
        
        self.avg_mean_diff = pd.DataFrame({pw.idx: pw.avg_diff() for pw in self.paired_windows})

    def __repr__(self):
        grades = [pw.score for pw in self.paired_windows]
        grades_formated = ("{:.2f}, " * len(grades)).format(*grades)
        return f'Subject windows grades=[{grades_formated}]'
    
    def visualize(self):
        plt.figure()
        self.avg_mean_diff.plot()
        plt.show()

    def get_data(self, train_num):
        prev_data = list(chain(*[w.get_data() for w in self.get_windows(train_num)]))

        last_data = self.paired_windows[train_num].watch_window.get_data(self.min_w)
        X = np.stack(prev_data + [last_data])
        y = self.paired_windows[train_num].score

        return X, y

    def get_windows(self, windows_num): return self.paired_windows[:windows_num]


class PairedWindows:
    def __init__(self, watch_window, regulate_window):
        def calc_score():
            mean_diff = self.watch_window.mean - self.regulate_window.mean
            joint_var = np.var(np.concatenate((self.watch_window.all_samples, self.regulate_window.all_samples)))
            return mean_diff / joint_var

        assert watch_window.idx == regulate_window.idx, f'indices mismatch: {watch_window.idx} != {regulate_window.idx}'
        self.idx = watch_window.idx
        self.watch_window: Window = watch_window
        self.regulate_window: Window = regulate_window
        self.score = calc_score()
        self.df = pd.DataFrame({**self.watch_window.series, **self.regulate_window.series})

        self.min_w = min(self.watch_window.w, self.regulate_window.w)

    def __repr__(self):
        return f'Windows #{self.idx}, score = {self.score:.4f}'

    def avg_diff(self):
        regulate_min = np.mean(self.regulate_window.np_mat[:, :self.min_w], axis=0)
        watch_min = np.mean(self.watch_window.np_mat[:, :self.min_w], axis=0)
        return np.asarray(watch_min - regulate_min, dtype=float)

    def get_data(self):
        return [w.get_data(self.min_w) for w in (self.watch_window, self.regulate_window)]


class Window:
    def __init__(self, idx, time_slots, window_type, bold_mat):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type

        self.voxels = {vox: Voxel(vox, bold_mat[(*vox), time_slots]) for vox in Subject.amyg_vox}
        self.series = ({str(vox) + window_type: pd.Series(self.voxels[vox].samples) for vox in self.voxels})
        self.np_mat = np.asarray([self.series[s].values for s in self.series])
        h, self.w = self.np_mat.shape
        self.all_samples = np.reshape(self.np_mat, h * self.w)
        self.mean = np.mean(self.all_samples)
        self.var = np.var(self.all_samples)

    def __repr__(self):
        return f"{self.window_type} window #{self.idx}, mean={self.mean:.2f}, var={self.var:.2f}"

    def get_data(self, min_w): return self.np_mat[:, :min_w]


class Voxel:
    def __init__(self, vox_coor, samples):
        self.vox_coor = vox_coor
        self.samples = samples
        self.mean = np.mean(samples)
        self.var = np.var(samples)

    def __repr__(self):
        return f"vox {self.vox_coor}, mean={self.mean:.2f}, var={self.var:.2f}"


def create_subject():
    subject_meta_data = SubjectMetaData(initial_delay=2,
                                        watch_on=[98, 151, 201, 251, 301],
                                        watch_duration=[23, 20, 20, 20, 20],
                                        regulate_on=[121, 171, 221, 271, 321],
                                        regulate_duration=[20, 20, 20, 20, 20],
                                        bold_mat_path='raw_data/BOLD_test.mat')

    subject = Subject(subject_meta_data)

    pickle.dump(subject, open('subject.pckl', 'wb'))
    return subject


def load_subject():
    subject = pickle.load(open('subject.pckl', 'rb'))
    return subject


if __name__ == '__main__':
    t = create_subject()
    print('subject created')
