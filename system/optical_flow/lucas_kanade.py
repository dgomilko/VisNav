import cv2
import numpy as np
from optical_flow.optical_flow import OpticalFlow

class LucasKanade(OpticalFlow):
	feature_params = {
		'maxCorners': 100,
		'qualityLevel': 0.3,
		'minDistance': 7,
		'blockSize': 7
	}
	
	def __init__(self, windows: list[np.ndarray]):
		self.__reset(windows)
		self.__good_new_arr = list()
		self.__prev = [0] * len(windows)

	def pre_update(self, windows: list[np.ndarray]) -> None:
		if self.count != 10: return
		self.__reset(windows)
		
	def get_flows(
			self,
			template: list[np.ndarray],
			target: list[np.ndarray]
		) -> list[tuple[int]]:
			flows = list()
			for i, (win, prev_win, p) in enumerate(zip(template, target, self.old_points)):
				flow, points = self.__lucas_kanade(prev_win, win, p, i)
				flows.append(flow)
				self.__good_new_arr.append(points)
			return flows
	
	def post_update(self) -> None:
		self.old_points = [good.reshape(-1, 1, 2) for good in self.__good_new_arr]
		self.count += 1

	def __lucas_kanade(
			self,
			template: np.ndarray,
			target: np.ndarray,
			points: np.ndarray,
			wnd_idx: int
		) -> tuple[tuple[int], np.ndarray]:
			no_points = points is None or not len(points)
			p_new, status = cv2.calcOpticalFlowPyrLK(target, template, points, None)[:2] \
				if not no_points else [None, []]
			if p_new is not None:
				good_new_points = p_new[status == 1]
				good_old_points = points[status == 1]
				diff = good_new_points - good_old_points
			else:
				diff = list()
				good_new_points = self.__good_new_arr[wnd_idx]
			no_prev = type(self.__prev[wnd_idx]) == list and len(self.__prev[wnd_idx])
			vector = np.mean(diff, axis=0) if len(diff) else \
				self.__prev[wnd_idx] if no_prev else [0, 0]
			self.__prev[wnd_idx] = vector
			return vector, good_new_points
	
	def __reset(self, windows) -> None:
		self.old_points = [cv2.goodFeaturesToTrack(
			prev,
			mask=None,
			**LucasKanade.feature_params
		) for prev in windows]
		self.count = 0
