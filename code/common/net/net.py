class Net(object):

  def __init__(self, **kwargs):
    raise NotImplementedError

  def inference(self, data):
    raise NotImplementedError

  def loss(self, layers, labels):
    raise NotImplementedError

  def metric(self, layers, labels):
    raise NotImplementedError
