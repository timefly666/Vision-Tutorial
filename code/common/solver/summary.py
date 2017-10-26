import os


class Summary(object):

  def __init__(self, **kwargs):
    raise NotImplementedError

  def accumulate(self, summary, step):
    raise NotImplementedError

  def summarize(self):
    raise NotImplementedError


class PrintSummary(Summary):

  def __init__(self, **kwargs):
    self.file_path = kwargs['file_path']
    self.base_dir = os.path.dirname(os.path.abspath(self.file_path))
    if not os.path.exists(self.base_dir):
      os.mkdir(self.base_dir)
    if os.path.exists(self.file_path):
      os.remove(self.file_path)

  def accumulate(self, summary, step):
    label = summary[0]
    predict = summary[1]
    with open(self.file_path, "a") as f:
      for gt, score in zip(label, predict):
        f.write("{}\t{}\n".format(gt[0], score[0]))

  def summarize(self):
    return
