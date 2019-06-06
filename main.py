# %%
import scipy.io as sio
import numpy as np
from functools import reduce
import pickle
from config import SubjectMetaData

class Subject:
    def __init__(self, meta_data: SubjectMetaData):
        def gen_windows(window_type):
            times_list = meta_data.watch_times if window_type == 'watch' else meta_data.regulate_times
            return map(lambda w: Window(*w, window_type, self.amyg_vox, self.bold), enumerate(times_list))

        self.meta_data = meta_data
        self.roi = np.where(sio.loadmat(meta_data.roi_mat_path)['ans'])
        self.amyg_vox = list(zip(*self.roi)) #  list(roi[0], roi[1], roi[2])
        self.bold = sio.loadmat(meta_data.bold_mat_path)['ans']

        self.paired_windows = list(map(PairedWindows, gen_windows('watch'), gen_windows('regulate')))
        

class PairedWindows:
    def __init__(self, watch_window, regulate_window):
        self.watch_window:Window = watch_window
        self.regulate_window:Window = regulate_window
        self.score = calc_window_score()
        
    def calc_window_score(self):
    mean_diff = self.watch_window.mean - self.regulate_window.mean
    joint_var = np.var(np.concatenate((self.watch_window.all_samples, self.regulate_window.all_samples)))
    return mean_diff / joint_var


class Window:
    def __init__(self, idx, time_slots, window_type, amyg_vox, bold_mat):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type

        self.voxels = {vox: Voxel(vox, bold_mat[(*vox), time_slots]) for vox in amyg_vox}
        self.all_samples = reduce(lambda x, y: np.concatenate((x, y)), map(lambda vox: vox.samples, self.voxels.values()))
        self.mean = np.mean(self.all_samples)
        self.var = np.var(self.all_samples)

    def __str__(self):
        return f"{self.window_type} window #{self.idx}, mean={self.mean:.2f}, var={self.var:.2f}"


class Voxel:
    def __init__(self, vox_coor, samples):
        self.vox_coor = vox_coor
        self.samples = samples
        self.mean = np.mean(samples)
        self.var = np.var(samples)

    def __str__(self):
        return f"vox {self.vox_coor}, mean={self.mean:.2f}, var={self.var:.2f}"


def main():
    subject_meta_data = SubjectMetaData(initial_delay=2,
                                        watch_on = [1, 54, 104, 154, 204],
                                        watch_duration = [23, 20, 20, 20, 20],
                                        regulate_on = [24, 74, 124, 174, 224],
                                        regulate_duration = [20, 20, 20, 20, 20],
                                        roi_mat_path='raw_data/roi.mat',
                                        bold_mat_path='raw_data/BOLD.mat')
                                        
    subject = Subject(subject_meta_data)
    
    pickle.dump(subject, open('windows.pckl', 'wb'))


if __name__ == '__main__':
    main()
