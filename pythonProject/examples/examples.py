import numpy as np
import matplotlib.pyplot as plt
from tol_func import Tol
from tol_func_mod import Tol_with_cost
import graw
from solver import Solve
from optimization_methods import ralgb5_with_proj, ralgb5

def example1():
  """
  Пример расчета методом проекции градиента
  и визуализации результата
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
  quantity_exp = 6
  lb = np.ones(quantity_exp * 2)
  ub = 2 * np.ones(quantity_exp * 2)
  quantity_starts = 30

  # Запускаем мультистарт
  res = solver.multistart(lb, ub, quantity_starts, ralgb5_with_proj ,return_all_data=False)
  print(*res, sep ="\n")

  # Рисуем результаты
  fx = lambda a, b, x: np.exp(-x * b) @ (a)

  graw.draw_interval(fx, res[1], res[2], x_lb, x_ub, y_lb, y_ub)
  plt.show()

def example2():
  """
  В этом примере тестируется решение оптимизационной задачи методом штрафной функции.
  Для большей сложности мы используем запуск из области не относящейся к допусковому множеству задачи
  """
  y_mid = np.array([2.185, 1.475, 1.2075, 0.98, 0.85, 0.7275])
  y_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.15
  y_lb = y_mid - y_rad
  y_ub = y_mid + y_rad
  x_mid = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
  x_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1
  x_lb = x_mid - x_rad
  x_ub = x_mid + x_rad


  quantity_exp = 6
  # стоимость выхода из области определения для a
  cost_a = 30.0 *np.ones(quantity_exp)
  # стоимость выхода из области определения для b
  cost_b = 30.0 *np.ones(quantity_exp)
  # класс распознающего функционала
  tol1 = Tol_with_cost(x_lb, x_ub, y_lb, y_ub, cost_a , cost_b)

  # Передаём функционал в солвер
  solver = Solve(tol1)

  #Задаём область из которой выбираются начальные приближения для мультистартов
  lb =  -2 * np.ones(quantity_exp * 2)
  ub = 0 * np.ones(quantity_exp * 2)
  quantity_starts = 30

  # Запускаем мультистарт
  res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)
  print(*res, sep="\n")

  # Рисуем лучший результат
  fx = lambda a, b, x: np.exp(-x * b) @ (a)

  graw.draw_interval(fx, res[1], res[2], x_lb, x_ub, y_lb, y_ub)
  plt.show()