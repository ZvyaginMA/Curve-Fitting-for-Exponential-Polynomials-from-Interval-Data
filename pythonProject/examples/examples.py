import numpy as np
import matplotlib.pyplot as plt
from tol_func import Tol
import graw
from solver import Solve

def example_1():
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