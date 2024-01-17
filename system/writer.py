import csv

class Writer:
  def __init__(self, filename: str):
    self.__csv_file = open(filename, 'w')
    self.__writer = csv.writer(self.__csv_file)
    titles = ['top', 'bottom', 'left', 'right', 'center']
    header = list()
    for title in titles:
      header.extend([f'vx_{title}', f'vy_{title}'])
    self.append_stats(header)

  def append_stats(self, data: list[str | float]) -> None:
    self.__writer.writerow(data)

  def __del__(self):
    self.__csv_file.close()
