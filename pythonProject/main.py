from examples.examples import *
from optimization_methods import *

def AEM():
  def calcfg(x):
    return np.linalg.norm(x), 2 * x

  met = Accelerated_ellipsoid_method()
  met.calc(calcfg, np.array([1.0, 1.0]), maxiter=10)



if __name__ == '__main__':
  def calcfg(x):
    a = np.array([1, 0.1])
    return np.linalg.norm(x * a), 2 * x * a
  opt = EMShor()
  print(opt.calc(calcfg, np.array([1.0, 2.0]), rad= 10))



