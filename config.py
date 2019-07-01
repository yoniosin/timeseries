from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class LearnerMetaData:
    latent_vector_size: int
    loss_lambda: float
    train_ratio: float = 0.8
    train_windows: int = 2

    def __post_init__(self):
        assert 0 < self.train_ratio < 1
        assert 0 < self.train_windows < 5
        meta_dict = json.load(open('PreProcess/meta.txt', 'r'))
        self.total_subject = meta_dict['subjects_num']
        self.min_w = meta_dict['min_w']
        self.voxels_num = meta_dict['voxels_num']
        self.in_channels = self.train_windows * 2 + 1

    def to_json(self): return asdict(self)


@dataclass
class SubjectMetaData:
    subject_name: str
    watch_on: List[int]
    watch_duration: List[int]
    regulate_on: List[int]
    regulate_duration: List[int]
    initial_delay: int = 2

    def gen_time_range(self, on, duration): return list(range(on + self.initial_delay, on + duration))

    def __post_init__(self):
        self.watch_times = map(self.gen_time_range, self.watch_on, self.watch_duration)
        self.regulate_times = map(self.gen_time_range, self.regulate_on, self.regulate_duration)
