class Dataset(object):

  def __init__(self, **kargs):
    raise NotImplementedError

  def batch(self):
    raise NotImplementedError

  def shape(self):
    raise NotImplementedError
