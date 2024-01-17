import numpy as np
from abc import ABC, abstractmethod

class OpticalFlow(ABC):
	@abstractmethod
	def pre_update(self, windows: list[np.ndarray]) -> None:
		pass

	@abstractmethod
	def get_flows(self,
			template: list[np.ndarray],
			target: list[np.ndarray]
		) -> list[tuple[int]]:
			pass

	@abstractmethod
	def post_update(self) -> None:
		pass
