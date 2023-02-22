import unittest
from tests.tests_optimization import *
from examples.examples import *
from optimization_methods import *

def AEM():
  def calcfg(x):
    return np.linalg.norm(x), 2 * x

  met = Accelerated_ellipsoid_method()
  met.calc(calcfg, np.array([1.0, 1.0]), maxiter=10)



if __name__ == '__main__':
  example3()



