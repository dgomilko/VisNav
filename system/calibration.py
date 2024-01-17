import numpy as np
import cv2
import os

class Calibration:
	def __init__(
		self,
		load_params: bool = False,
		save_params: bool = False,
		dataset_path: str = './c_images',
		params_path: str = './calib.npz',
		chess_dims: tuple[int] = (6, 9)
	):
			self.__params = self.__load(params_path) if load_params else \
				self.__calibrate_from_dataset(dataset_path, chess_dims)
			if save_params:
				mtx, dist, rvecs, tvecs = self.__params
				np.savez('calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
 
	def undistort(self, img: np.ndarray) -> np.ndarray:
		mtx, dist = self.__params[:2]
		h, w = img.shape[:2]
		new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
		dst = cv2.undistort(img, mtx, dist, None, new_mtx)
		x, y, w, h = roi
		return dst[y:y + h, x:x + w]

	def __load(self, path: str) -> list[np.ndarray]:
		data = np.load(path)
		return [data[item] for item in data.files]

	def __calibrate_from_dataset(self, path: str, dims: tuple[int]) -> list[np.ndarray]:
		a, b = dims
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
		objp = np.zeros((a * b, 3), np.float32)
		objp[:,:2] = np.mgrid[0:b, 0:a].T.reshape(-1, 2)
		p_3d, p_2d = (list(), list())
		images = [f'{path}/{file}' for file in os.listdir(path)]
		for fname in images:
			gray = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (b, a), None)
			if not ret: continue
			p_3d.append(objp)
			sub_pix = cv2.cornerSubPix(gray,corners, (11, 11), (-1,-1), criteria)
			p_2d.append(sub_pix)
		return cv2.calibrateCamera(p_3d, p_2d, gray.shape[::-1], None, None)[1:]
