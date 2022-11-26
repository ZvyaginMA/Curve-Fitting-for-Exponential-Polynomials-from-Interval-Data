import numpy as np
import matplotlib.pyplot as plt
from tol_func import Tol
from tol_func_mod import Tol_with_cost
from optimization_methods import ralgb5, ralgb5_with_proj
import graw




def main3():
  y_mid = np.array([2.185, 1.475, 1.2075, 0.98, 0.85, 0.7275])
  y_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5

  y_lb = y_mid - y_rad
  y_ub = y_mid + y_rad
  x_mid = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
  x_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1

  x_lb = x_mid - x_rad
  x_ub = x_mid + x_rad
  tol1 = Tol(x_lb, x_ub, y_lb, y_ub)

  def calcfg2(x):
    a, b = x[:len(x) // 2], x[len(x) // 2:]
    return - tol1.tol_value(a, b), np.concatenate(
      [- tol1.dTda(a, b), - tol1.dTdb(a, b)])
  #рисуем 3 графика для трех начальных приближений
  for i in range(3):
    plt.subplot(3, 1, i + 1)
    xr, fr, nit, ncalls, ccode = ralgb5(calcfg2,
                                                     np.array([2.5, 2.9, 3.9, 2.9, 3.9, 2.9]) + 0.3 * i)  # 2.9, 3.9,
    a, b = xr[:len(xr) // 2], xr[len(xr) // 2:]
    print(f"a = {a} \nb = {b} \ntolval = {-fr}")
    fx = lambda a, b, x: np.exp(-x * b) @ (a)
    graw.draw_interval(fx, a, b, x_lb, x_ub, y_lb, y_ub)
    print(fr, nit, ncalls, ccode)
  # graw_for_two_b(a)
  plt.show()

class Solve:
  def __init__(self, tol):
    self.tol = tol

  def multistart(self, lb, ub, quantity_starts, return_all_data=True):
    def calcfg(x):
      a, b = x[:len(x) // 2], x[len(x) // 2:]
      return - tol1.tol_value(a, b), np.concatenate(
        [- tol1.dTda(a, b), - tol1.dTdb(a, b)])
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
    """
    print(f"a = {a} \nb = {b} \ntolval = {-fr}")
    fx = lambda a, b, x: np.exp(-x * b) @ (a)
    if True: #(-fr > 0):
      graw.draw_interval(fx, a, b, x_lb, x_ub, y_lb, y_ub)
      print(fr, nit, ncalls, ccode)
    """

if __name__ == '__main__':
  """
  Пример расчета и визуализации данных
  """
  # Задание входных данных
  y_mid = np.array([2.185, 1.475, 1.2075, 0.98, 0.85, 0.7275])
  y_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.15
  y_lb = y_mid - y_rad
  y_ub = y_mid + y_rad
  x_mid = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
  x_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1
  x_lb = x_mid - x_rad
  x_ub = x_mid + x_rad
  # класс распознающего функционала
  tol1 = Tol(x_lb, x_ub, y_lb, y_ub)

  # Передаём функционал в солвер
  solver = Solve(tol1)
  quantity_exp = 2 
  lb = np.ones(quantity_exp * 2)
  ub = 2 * np.ones(quantity_exp * 2)
  quantity_starts = 100

  # Запускаем мультистарт
  res = solver.multistart(lb, ub, quantity_starts, return_all_data=False)
  print(*res, sep ="\n")

  # Рисуем результаты
  fx = lambda a, b, x: np.exp(-x * b) @ (a)
  if True:  # (-fr > 0):
    graw.draw_interval(fx, res[1], res[2], x_lb, x_ub, y_lb, y_ub)
  plt.show()

