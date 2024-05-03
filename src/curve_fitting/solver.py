import numpy as np
from joblib import Parallel, delayed

class Solver:
  def __init__(self, tol):
    self.tol = tol

  def multistart(self,  lb, ub, quantity_starts, method, return_all_data=False):
      
    best_res = {"func_val":-float("inf")}
    X = np.zeros((quantity_starts, len(lb)), dtype=np.float32)
    for i in range(len(lb)):
      X[:, i] = (np.random.uniform(lb[i], ub[i], (quantity_starts)))
    
    def process(i):
        return self.tol.optimization(method, X[i])
        
    results = Parallel(n_jobs=np.min([np.max([quantity_starts //2,1]) , 10]))(delayed(process)(i) for i in range(len(X)))
    positive_res = list(filter(lambda x: x["func_val"] >= 0, results))
    best_res = max(results, key= lambda x: x["func_val"])

    if return_all_data:
      return {"best_res" :best_res, "all_res" : positive_res}
    else:
      return {"best_res" :best_res}

