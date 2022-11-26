import numpy as np
from optimization_methods import ralgb5, ralgb5_with_proj


class Solve:
  def __init__(self, tol):
    self.tol = tol

  def multistart(self,  lb, ub, quantity_starts, return_all_data=True):
    def calcfg(x):
      a, b = x[:len(x) // 2], x[len(x) // 2:]
      return - self.tol.tol_value(a, b), np.concatenate(
        [- self.tol.dTda(a, b), - self.tol.dTdb(a, b)])
    res = []
    best_res = (-float("inf"),0)
    X = np.zeros((quantity_starts, len(lb)), dtype=np.float32)
    for i in range(len(lb)):
      X[:, i] = (np.random.uniform(lb[i], ub[i], (quantity_starts)))

    for x in X:
      res.append(self.optimization(calcfg, x))
      if res[-1][0] > best_res[0]:
        best_res = res[-1]

    if return_all_data == True:
      return res
    else:
      return best_res

  def optimization(self, calcfg, x_0):
    xr, fr, nit, ncalls, ccode = ralgb5_with_proj(calcfg, x_0) #2.9, 3.9,
    a, b = xr[:len(xr) // 2], xr[len(xr) // 2:]
    return ( - fr, a, b,ccode )
