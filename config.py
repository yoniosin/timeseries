from dataclasses import dataclass
from collections import namedtuple


window_times = namedtuple('window_times', ['on', 'duration'])


@dataclass
class SubjectMetaData:
	initial_dealy:int
	watch_on: List[int]
	watch_duration: List[int]
	regulate_on: List[int]
	regulate_duration: List[int]
	roi_mat_path: str
	bold_mat_path: str
	
	def __post_init__(self):
		self.watch_times = list(map(window_times, self.watch_on, self.watch_duration))
		self.regulate_times = list(map(window_times, self.regulate_on, self.regulate_duration))
	
