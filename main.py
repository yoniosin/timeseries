# %%
import scipy.io as sio
import numpy as np
from functools import reduce
import pickle


def calc_window_score(watch_wind, regulate_wind):
    mean_diff = watch_wind.mean - regulate_wind.mean
    joint_var = np.var(np.concatenate((watch_wind.all_samples, regulate_wind.all_samples)))
    return mean_diff / joint_var


class Window:
    def __init__(self, idx, time_slots, window_type):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type

        self.voxels = {vox: Voxel(vox, bold[(*vox), time_slots]) for vox in amyg_vox}
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


# %% load ROI mat

roi = np.where(sio.loadmat('raw_data/roi.mat')['ans'])
amyg_vox = list(zip(roi[0], roi[1], roi[2]))
bold = sio.loadmat('raw_data/BOLD.mat')['ans']


def main():
    initial_delay = 2

    def gen_times(on, duration):
        return list(range(on + initial_delay, on + duration))

    w_on = [1, 54, 104, 154, 204]
    w_duration = [23, 20, 20, 20, 20]
    w_times = map(gen_times, w_on, w_duration)

    r_on = [24, 74, 124, 174, 224]
    r_duration = [20, 20, 20, 20, 20]
    r_times = map(gen_times, r_on, r_duration)

    windows = {'watch': [Window(idx, time_slots, 'watch') for idx, time_slots in enumerate(w_times)],
               'regulate': [Window(idx, time_slots, 'regulate') for idx, time_slots in enumerate(r_times)]}
    windows_score = list(map(calc_window_score, windows['watch'], windows['regulate']))
    pickle.dump(windows, open('windows.pckl', 'wb'))


if __name__ == '__main__':
    main()
