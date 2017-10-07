class Solver(object):

  def __init__(self, dataset, net, **kargs):
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  def test(self):
    raise NotImplementedError

  def inference(self):
    raise NotImplementedError
