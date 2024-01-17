import cv2
import numpy as np
from writer import Writer
from options import Options
from collections import deque
from optical_flow.lucas_kanade import LucasKanade
from mav_connection import MavConnection
from optical_flow.cross_correlation import Correlation

class FrameProcessor:
  OF_types = {
    'corr': Correlation,
    'lk': LucasKanade,
  }

  def __init__(self, capture: cv2.VideoCapture, options: Options, coords: np.ndarray):
    self.__capture = capture
    self.__coords = coords
    self.__options = options
    # Connect via MVLink protocol
    self.__conn = MavConnection()
    self.__method = None
    self.__buffers = [deque(maxlen=options.fps) for _ in range(len(coords))]
    self.__writer = Writer(options.output_path)
    self.__modes = {
    	'speed': self.__get_speed,
    	'height': self.__get_height,
  	}

  # Convert parts of a frame to grayscale
  def gray_windows(self, frame, coords) -> list[np.ndarray]:
    wnd = [
    cv2.cvtColor(frame[y_st:y_fin, x_st:x_fin], cv2.COLOR_BGR2GRAY)
      for ((x_st, y_st), (x_fin, y_fin)) in coords
    ]
    if not self.__method:
      OF_type = self.__options.optical_flow_type
      self.__method = FrameProcessor.OF_types[OF_type](wnd)
    else:
      self.__method.pre_update(wnd)
    return wnd

  # Calculate motion vectors for each frame window
  def calc_flows(
      self,
      prev_frame: np.ndarray
    ) -> tuple[list[np.ndarray], np.ndarray]:
    frame = self.__capture.read()[1]
    if frame is None: return
    # Convert frame windows to grayscale
    cur_gray = self.gray_windows(frame, self.__coords)
    # Get motion vector for each frame window
    flows = self.__method.get_flows(prev_frame, cur_gray)
    # Store values in buffer
    for i, flow in enumerate(flows):
      self.__buffers[i].append(flow)
    return cur_gray, frame

  # Calculate mean total distance or height for each window 
  def sum_distances(self) -> None:
    total = self.__modes[self.__options.mode]()
    self.__writer.append_stats(np.array(total).flatten())
    self.__method.post_update()
  
  def __get_speed(self) -> list[tuple[float]]:
    alt_mm = self.__conn.receive_gps_position()['relative_alt']
    mm_per_pixel = alt_mm * self.__options.pixel_coef
    total = [
      np.sum(np.array(buff), axis=0) * mm_per_pixel for buff in self.__buffers
    ]
    return total
  
  def __get_height(self) -> list[float]:
    pos = self.__conn.receive_gps_position()
    vx = pos['vx']
    vy = pos['vy']
    speed_mm = self.__get_dist(vx, vy) * 10
    heights = list()
    for buffer in self.__buffers:
      buff = np.array(buffer)
      speed = self.__get_dist(*np.sum(buff, axis=0))
      heights.append(speed_mm / (speed * self.__options.pixel_coef))
    return heights

  __get_dist = lambda x, y: (x * x + y * y) ** 0.5
