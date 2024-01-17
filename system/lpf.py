import matplotlib.pyplot as plt
import numpy as np

def low_pass_filter(
    data: np.ndarray,
    freq_limit: int = 15,
    sample: float = 0.01
  ) -> np.ndarray:
    result = list()
    out = data[0]

    for i in range(data.size):
      out += freq_limit * (data[i] - out) * sample
      result.append(out)

    return result

def real_time_lpf(
    data: np.ndarray,
    freq_limit: int = 10,
    sample: float = 0.005
  ) -> list[float]:
    return [low_pass_filter(
      data[:i + 1] if i < 100 else data[i - 100:i + 1],
      freq_limit, sample
    )[-1] for i in range(data.size)]
