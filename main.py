# %%
import scipy.io as sio
import numpy as np
from functools import reduce
import pickle
from collections inport namedtuple

window_times = namedtuple('window_times', ['on', 'duration'])

class Subject:
    amyg_vox = None
    bold = None
    
    def __init__(self, watch_times: window_times, regulate_times: window_times, initial_delay, roi_location):
        def gen_windows(times_list, window_type):
            return map(lambda w: gen_single_window(*w, window_type), times_list)
            def gen_single_window(idx, window_times, window_type):
                return Window(idx, list(range(window_times.on + initial_delay, window_times.on + window_times.duration)), window_type)

        self.roi_location = roi_location
        self.roi = np.where(sio.loadmat(roi_location)['ans'])
        Subject.amyg_vox = list(zip(roi[0], roi[1], roi[2])) #  list(zip(*roi))
        Subject.bold = sio.loadmat('raw_data/BOLD.mat')['ans']

        
        self.initial_delay = initial_delay
        self.watch_times = watch_times
        self.regulate_times = regulate_times
        self.paired_windows = list(map(PairedWindows, gen_windows(watch_times, 'watch'), gen_windows(watch_times, 'regulate')))
        

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
    def __init__(self, idx, time_slots, window_type):
        self.idx = idx
        self.time = time_slots
        self.window_type = window_type

        self.voxels = {vox: Voxel(vox, Subject.bold[(*vox), time_slots]) for vox in Subject.amyg_vox}
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
    initial_delay = 2

    w_on = [1, 54, 104, 154, 204]
    w_duration = [23, 20, 20, 20, 20]
    w_times = map(window_times, w_on, w_duration)

    r_on = [24, 74, 124, 174, 224]
    r_duration = [20, 20, 20, 20, 20]
    r_times = map(window_times, r_on, r_duration)

    subject = Subject(w_times, r_times, initial_delay, 'raw_data/roi.mat')
    
    pickle.dump(subject, open('windows.pckl', 'wb'))


if __name__ == '__main__':
    main()
