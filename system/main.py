import cv2
import numpy as np
from options import get_config
from frame_processor import FrameProcessor

def analyze_videostream():
  pass 

WIN_NAME = 'Video'
CONFIG_PATH = '../config.json'

# Resize window to fit the screen
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 1728, 972)

# Load execution options from json file
options = get_config(CONFIG_PATH)
# Start video capture frame-by-frame
cap = cv2.VideoCapture(options.video_source)
# Get frame-per-seconds from the capture properties
options.fps = int(cap.get(cv2.CAP_PROP_FPS))
# Get the first frame from the capture
first_frame = cap.read()[1]
# Get the video dimentions
dims = first_frame.shape[:2]
height, width = dims
# Calculate pixel size in millimetres
options.pixel_coef = options.scale_coef / width

w_size = options.win_size
pad = options.padding_px

ret, first_frame = cap.read()
dims = first_frame.shape[:2]
y, x = [int(x / 2 - w_size / 2) for x in dims]

coords = np.array([
  [(x, pad), (x + w_size, pad + w_size)], # top
  [(x, height - (pad + w_size)), (x + w_size, height - pad)], # bottom
  [(pad, y), (pad + w_size, y + w_size)], # left
  [(width - (pad + w_size), y), (width - pad, y + w_size)], # right
  [(x, y), (x + w_size, y + w_size)], # center
])

frame_proc = FrameProcessor(cap, options, coords)
# Get windows for the first frame, converted to grayscale
prev_gray = frame_proc.gray_windows(first_frame, coords)
frame_count = 0

while True:
  try:
    # Get motion vectors for current frame
    result = frame_proc.calc_flows(prev_gray)
    # Break the loop if the video is ended
    if result is None: break
    cur_gray, cur_frame = result
    # Display current frame in the window
    cv2.imshow(WIN_NAME, cur_frame)
    # Find total estimated distance
    if frame_count == options.fps:
      frame_proc.sum_distances()
      # Reset the counter
      frame_count = 0
    prev_gray = cur_gray.copy()
    c = cv2.waitKey(1)
    esc_pressed = c == 27
    frame_count += 1
    if esc_pressed: break
  except KeyboardInterrupt: break

cap.release()
cv2.destroyAllWindows()
