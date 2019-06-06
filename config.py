from dataclasses import dataclass
from collections import namedtuple


@dataclass
class SubjectMetaData:
	initial_delay:int
	watch_on: List[int]
	watch_duration: List[int]
	regulate_on: List[int]
	regulate_duration: List[int]
	roi_mat_path: str
	bold_mat_path: str
	
	def gen_time_range(on, duration): 
		start = on + self.initial_delay
		return list(range(start, start + duration))
	
	def __post_init__(self):
		self.watch_times = map(gen_time_range, self.watch_on, self.watch_duration)
		self.regulate_times = map(gen_time_range, self.regulate_on, self.regulate_duration)
	
