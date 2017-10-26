class Solver(object):

  def __init__(self, dataset, net, **kargs):
    raise NotImplementedError

  def train(self):
    raise NotImplementedError

  def eval(self):
    raise NotImplementedError

  def predict(self):
    raise NotImplementedError
