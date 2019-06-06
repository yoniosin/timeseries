from dataclasses import dataclass

@dataclass
class SubjectMetaData:
	initial_dealy:int
	watch_on: List[int]
	watch_duration: List[int]
	regulate_on: List[int]
	regulate_duration: List[int]
	roi_mat_path: str
	bold_mat_path: str
