from dataclasses import dataclass
from typing import List


@dataclass
class SubjectMetaData:
    initial_delay: int
    watch_on: List[int]
    watch_duration: List[int]
    regulate_on: List[int]
    regulate_duration: List[int]
    bold_mat_path: str
    bold_mat_name: str = 'y'

    def gen_time_range(self, on, duration): return list(range(on + self.initial_delay, on + duration))

    def __post_init__(self):
        self.watch_times = map(self.gen_time_range, self.watch_on, self.watch_duration)
        self.regulate_times = map(self.gen_time_range, self.regulate_on, self.regulate_duration)
