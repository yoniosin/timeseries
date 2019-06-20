import scipy.io as sio
import numpy as np
import pickle
from config import SubjectMetaData
from itertools import chain
import mat4py
from oct2py import Oct2Py
import gdown
import re


class TrialData:
    def __init__(self, protocol_path):
        protocols = mat4py.loadmat(protocol_path)['Protocols']
        keep_elements = ['subject', 'TestWatchOnset', 'TestWatchDuration', 'TestRegulateOnset', 'TestRegulateDuration']
        p_per_subject = zip(*(map(lambda a: protocols[a], keep_elements)))
        protocols = {protocol[0]: protocol[1:] for protocol in p_per_subject}

        def create_subject(link):
            def download_bold_mat():
                needed = re.search(r'file/d/(.*)/view', link).group(1)
                new_link = ''.join(['https://drive.google.com/uc?id=', needed, '&export=download'])
                gdown.download(new_link, 'bold_mat.mat', quiet=False)
                mat = oc.convert_img('bold_mat.mat')
                s_id = re.search(r'(Subject\d*)', open('subject_name.txt', 'r').read()).group(1)

                return mat, s_id

            bold_mat, subject_id = download_bold_mat()
            protocol = protocols[subject_id]
            return Subject(SubjectMetaData(subject_id, *protocol), bold_mat)

        with open('download_links.txt') as fp:
            self.subjects = [create_subject(link) for link in fp]

    def add_subjects(self, subjects):
        Subject.update_min(min(map(lambda s: s.min_w, subjects)))
        self.subjects += subjects


class Subject:
    roi = np.where(sio.loadmat('raw_data/roi.mat')['ans'])
    amyg_vox = list(zip(*roi))
    shared_min_w = np.inf

    @classmethod
    def update_min(cls, candidate): cls.shared_min_w = min(cls.shared_min_w, candidate)

    def __init__(self, meta_data: SubjectMetaData, bold_mat):
        def gen_windows(window_type):
            times_list = meta_data.watch_times if window_type == 'watch' else meta_data.regulate_times
            return map(lambda w: Window(*w, window_type, self.bold), enumerate(times_list))

        self.meta_data = meta_data
        self.name = meta_data.subject_name
        self.bold = bold_mat
        self.paired_windows = list(map(PairedWindows, gen_windows('watch'), gen_windows('regulate')))

        pickle.dump(self, open(f'../Data/{self.name}.pkl', 'wb'))

    def __repr__(self):
        grades = [pw.score for pw in self.paired_windows]
        grades_formatted = ("{:.2f}, " * len(grades)).format(*grades)
        return f'{self.name} windows grades=[{grades_formatted}]'
    
    def get_data(self, train_num):
        prev_data = list(chain(*[w.get_data() for w in self.get_windows(train_num)]))

        last_pw = self.paired_windows[train_num]
        last_data = last_pw.watch_window.get_data()
        X = np.stack(prev_data + [last_data])
        y = last_pw.score

        return X, y

    def get_windows(self, windows_num): return self.paired_windows[:windows_num]


class PairedWindows:
    def __init__(self, watch_window, regulate_window):
        def calc_score():
            mean_diff = self.watch_window.mean - self.regulate_window.mean
            joint_var = np.var(np.concatenate((self.watch_window.all_samples, self.regulate_window.all_samples)))
            map(lambda w: delattr(w, 'all_samples'), (watch_window, regulate_window))
            return mean_diff / joint_var

        assert watch_window.idx == regulate_window.idx, f'indices mismatch: {watch_window.idx} != {regulate_window.idx}'
        self.idx = watch_window.idx
        self.watch_window: Window = watch_window
        self.regulate_window: Window = regulate_window
        self.score = calc_score()

    def __repr__(self):
        return f'Windows #{self.idx}, score = {self.score:.4f}'

    def get_data(self):
        return [w.get_data() for w in (self.watch_window, self.regulate_window)]


class Window:
    def __init__(self, idx, time_slots, window_type, bold_mat):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type

        self.voxels = {vox: Voxel(vox, bold_mat[(*vox), time_slots]) for vox in Subject.amyg_vox}
        self.np_mat = np.asarray([voxel.samples for voxel in self.voxels.values()])
        h, self.w = self.np_mat.shape
        Subject.update_min(self.w)
        self.all_samples = np.reshape(self.np_mat, h * self.w)
        self.mean = np.mean(self.all_samples)
        self.var = np.var(self.all_samples)

    def __repr__(self):
        return f"{self.window_type} window #{self.idx}, mean={self.mean:.2f}, var={self.var:.2f}"

    def get_data(self): return self.np_mat[:, :Subject.shared_min_w]


class Voxel:
    def __init__(self, vox_coor, samples):
        self.vox_coor = vox_coor
        self.samples = samples
        self.mean = np.mean(samples)
        self.var = np.var(samples)

    def __repr__(self):
        return f"vox {self.vox_coor}, mean={self.mean:.2f}, var={self.var:.2f}"


def load_trial_data():
    return pickle.load(open("raw_data/trial_data.pckl", "rb"))


if __name__ == '__main__':
    oc = Oct2Py()
    data = TrialData('raw_data/ProtocolBySub_new.mat')
    pickle.dump(data, open('raw_data/trial_data.pckl', 'wb'))
