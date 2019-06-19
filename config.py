from dataclasses import dataclass
from typing import List


prefix = 'dataMat.mat'
@dataclass
class LearnerMetaData:
    bold_mat_location: str
    train_ratio: float = 0.7
    train_windows: int = 2

    def __post_init__(self):
        assert 0 < self.train_ratio < 1
        assert 0 < self.train_windows < 5


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
        self.file_name = '_'.join([self.subject_name, prefix])
