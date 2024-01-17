import json
from dataclasses import dataclass

@dataclass
class Options:
  scale_coef: float
  win_size: int
  padding_px: int
  video_source: int | str = 0
  output_path: str = 'res.csv'
  optical_flow_type: str = 'corr'
  mode: str = 'speed'
  fps: int | None = None
  pixel_coef: float | None = None

  def __post_init__(self):
    if not self.optical_flow_type in ['corr', 'lk']:
      msg = 'Optical flow type must be either "corr" for correlation or "lk" for Lucas-Kanade'
      raise ValueError(msg)
    if not self.mode in ['speed', 'height']:
      msg = 'Mode type must be either "speed" or "height"'
      raise ValueError(msg)

def get_config(conf_path: str) -> Options:
  with open(conf_path, 'r') as f:
    entries = json.loads(f.read())
  return Options(**entries)
