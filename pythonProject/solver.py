import numpy as np

class Solve:
  def __init__(self, tol):
    self.tol = tol

  def multistart(self,  lb, ub, quantity_starts, method, return_curve_progress = False, return_all_data=False):
    def calcfg(x):
      a, b = x[:len(x) // 2], x[len(x) // 2:]
      return - self.tol.tol_value(a, b), np.concatenate(
        [- self.tol.dTda(a, b), - self.tol.dTdb(a, b)])
    all_res = []
    curv_progress = []
    best_res = (-float("inf"),0)
    X = np.zeros((quantity_starts, len(lb)), dtype=np.float32)
    for i in range(len(lb)):
      X[:, i] = (np.random.uniform(lb[i], ub[i], (quantity_starts)))

    for x in X:
      iter_res = self.optimization(calcfg, method, x)
      if iter_res[0] > best_res[0]:
        best_res = iter_res
      if return_all_data:
        all_res.append(iter_res)
      if return_curve_progress:
        curv_progress.append(best_res[0])


    if return_all_data:
      if curv_progress:
        return best_res, np.array(curv_progress), all_res
      else:
        return best_res, all_res
    else:
      if curv_progress:
        return best_res, np.array(curv_progress)
      else:
        return best_res

  def optimization(self, calcfg, method , x_0):
    xr, fr, nit, ncalls, ccode = method(calcfg, x_0) #2.9, 3.9,
    a, b = xr[:len(xr) // 2], xr[len(xr) // 2:]
    return ( - fr, a, b,ccode )


