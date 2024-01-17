import numpy as np
from scipy.signal import fftconvolve
from optical_flow.optical_flow import OpticalFlow

class Correlation(OpticalFlow):
	def __init__(self, windows: list[np.ndarray]):
		pass
		
	def get_flows(
			self,
			template: list[np.ndarray],
			target: list[np.ndarray]
		) -> list[tuple[int]]:
			zipped = zip(target, template)
			return [self.__cross_correlate(win, prev_win) for win, prev_win in zipped]
	
	def __cross_correlate(
			self,
			template: np.ndarray,
			target: np.ndarray,
			mode: str = 'full'
		) -> tuple[int]:
			templ_sh = template.shape
			more_dims = np.ndim(template) > np.ndim(target)
			template_larger = any(x > y for x, y in zip(templ_sh, target.shape))
			if more_dims or template_larger: return
			
			templ_norm, target_norm = [arr - np.mean(arr) for arr in (template, target)]
			a1 = np.ones(templ_sh)
			template_flipped = np.flipud(np.fliplr(templ_norm))
			numerator = fftconvolve(target_norm, template_flipped.conj(), mode=mode)
			
			target_fft = fftconvolve(np.square(target_norm), a1, mode=mode) - \
				np.square(fftconvolve(target_norm, a1, mode=mode)) / np.prod(templ_sh)
			target_fft[np.where(target_fft < 0)] = 0
			
			template_sum = np.sum(np.square(templ_norm))
			with np.errstate(divide='ignore',invalid='ignore'): 
				out = numerator / np.sqrt(target_fft * template_sum)
			out[np.where(np.logical_not(np.isfinite(out)))] = 0
			
			peak_idx = np.array(np.unravel_index(out.argmax(), out.shape))
			offsets = peak_idx - np.array(templ_sh)
			return (offsets[1], -offsets[0])
	
	def post_update(self) -> None:
		return
	
	def pre_update(self, windows: list[np.ndarray]) -> None:
		return
