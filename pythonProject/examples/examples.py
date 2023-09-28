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
  res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_curve_progress= False,return_all_data=False)
  print(*res, sep="\n")

  # Рисуем лучший результат
  fx = lambda a, b, x: np.exp(-x * b) @ (a)

  graw.draw_interval(fx, res[1], res[2], x_lb, x_ub, y_lb, y_ub)
  plt.show()

def example3():
  """
  Пример расчета методом штрафной функции
  и визуализации кривой оптимизации
  """
  # Задание входных данных
  y_mid = np.array([2.185, 1.475, 1.2075, 0.98, 0.85, 0.7275])
  y_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.2
  y_lb = y_mid - y_rad
  y_ub = y_mid + y_rad
  x_mid = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
  x_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1
  x_lb = x_mid - x_rad
  x_ub = x_mid + x_rad

  quantity_exp = 6
  lb = np.ones(quantity_exp * 2)
  ub = 2 * np.ones(quantity_exp * 2)
  quantity_starts = 10
  cost_a = 3.0 * np.ones(quantity_exp)
  # стоимость выхода из области определения для b
  cost_b = 3.0 * np.ones(quantity_exp)
  # класс распознающего функционала
  tol1 = Tol_with_cost(x_lb, x_ub, y_lb, y_ub, cost_a, cost_b)
  solver = Solve(tol1)

  # Запускаем мультистарт
  res, curve = solver.multistart(lb, ub, quantity_starts, ralgb5 ,return_curve_progress=True, return_all_data=False)
  print(*res, sep ="\n")

  fx = lambda a, b, x: np.exp(-x * b) @ (a)

  graw.draw_interval(fx, res[1], res[2], x_lb, x_ub, y_lb, y_ub)
  plt.show()

  plt.scatter(range(len(curve)), curve, c="r")
  plt.xlabel("Итерации мультистарта")
  plt.ylabel("Tol")
  plt.show()
  # Рисуем результаты


def example4():
  """
    Построение графика демонстрирующего невыпуклость при параметрах из примера 1 из курсовой
  """
  y_mid = np.array([2.185, 1.475, 1.2075, 0.98, 0.85, 0.7275])
  y_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.3
  y_lb = y_mid - y_rad
  y_ub = y_mid + y_rad
  x_mid = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
  x_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.4
  x_lb = x_mid - x_rad
  x_ub = x_mid + x_rad

  tol1 = Tol(x_lb, x_ub, y_lb, y_ub)

  a = np.array([1.7, 0.4])

  def F(U, V, a):
    z = np.zeros_like(U)
    for i in range(len(U)):
      for j in range(len(U[0])):
        z[i, j] = tol1.tol_value(a, np.array([U[i, j], V[i, j]]))
    return z

  u = np.linspace(0, 2.5, 100)
  v = np.linspace(0, 2.5, 100)

  U, V = np.meshgrid(u, v)
  z = F(U, V, a)
  cs = plt.contourf(U, V, z, 20)
  plt.show()


def example5():
  """
  Пример расчета методом штрафной функции
  и визуализации кривой оптимизации
  """
  # Задание входных данных
  y_mid = np.array([2.185, 1.475, 1.2075, 0.98, 0.85, 0.7275])
  y_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.35
  y_lb = y_mid - y_rad
  y_ub = y_mid + y_rad
  x_mid = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
  x_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1
  x_lb = x_mid - x_rad
  x_ub = x_mid + x_rad

  quantity_exp = 6
  lb = np.ones(quantity_exp * 2)
  ub = 2 * np.ones(quantity_exp * 2)
  quantity_starts = 10
  cost_a = 3.0 * np.ones(quantity_exp)
  # стоимость выхода из области определения для b
  cost_b = 3.0 * np.ones(quantity_exp)
  # класс распознающего функционала
  tol1 = Tol_with_cost(x_lb, x_ub, y_lb, y_ub, cost_a, cost_b)
  solver = Solve(tol1)

  # Запускаем мультистарт
  res, curve = solver.multistart(lb, ub, quantity_starts, ralgb5 ,return_curve_progress=True, return_all_data=False)
  print(*res, sep ="\n")

  fx = lambda a, b, x: np.exp(-x * b) @ (a)

  graw.draw_interval(fx, res[1], res[2], x_lb, x_ub, y_lb, y_ub)
  plt.show()

  plt.scatter(range(len(curve)), curve, c="r")
  plt.xlabel("Итерации мультистарта")
  plt.ylabel("Tol")
  plt.show()
  # Рисуем результаты


def example6():
    # Задание входных данных
  y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93, 0.77, 0.64, 0.53, 0.45, 0.38, 0.32, 0.27, 0.23, 0.20, 0.17, 0.15, 0.13, 0.11, 0.10, 0.09,0.08, 0.07, 0.06])
  y_rad = np.ones(24) * 0.005
  y_lb = y_mid - y_rad
  y_ub = y_mid + y_rad
  x_mid = np.array(range(0, 24, 1)) * 0.05
  x_rad = np.ones(24) * 0.0
  x_lb = x_mid - x_rad
  x_ub = x_mid + x_rad

  for i in range(12):
    print(f"{i + 1 } & {x_mid[i]}& {y_mid[i]} & {i + 1 + 12} & {x_mid[i+ 12]}& {y_mid[i+ 12]}")

  tol = Tol(x_lb, x_ub, y_lb, y_ub)
  a_t = np.array([0.0951,0.8607,1.5576 ])
  b_t = np.array([1.0, 3.0, 5.0])
  res1 = tol.tol_value(a_t, b_t)
  a_l = np.array([2.202, 0.305])
  b_l = np.array([4.45, 1.58])
  res2 = tol.tol_value(a_l, b_l)
  a_l = np.array(  [2.08404657e+00, 4.57120604e-12, 4.25673560e-01])
  b_l = np.array(  [4.60646509, 0.51699632, 1.85283244])
  res3 = tol.tol_value(a_l, b_l)
  print(res1, res2, res3)
  quantity_exp = 4
  lb = np.ones(quantity_exp * 2) * 0.000001
  ub = 20 *  np.ones(quantity_exp * 2)
  quantity_starts = 10
  cost_a = 10.0 * np.ones(quantity_exp)
  # стоимость выхода из области определения для b
  cost_b = 10.0 * np.ones(quantity_exp)
  # класс распознающего функционала
  tol1 = Tol_with_cost(x_lb, x_ub, y_lb, y_ub, cost_a, cost_b)
  solver = Solve(tol1)

  f = lambda x: np.array([0.0951, 0.8607, 1.5576]) @ np.exp(x * np.array([-1.0, -3.0, -5.0]))
  fm = [f(x) for x in x_mid]

  """
  0.0016801321866205532
[2.08404657e+00 4.57120604e-12 4.25673560e-01]
[4.60646509 0.51699632 1.85283244]
1

0.0016801321866264374
[2.08404657 0.42567356]
[4.60646509 1.85283244]
1

0.0016801321866166674
[6.68499642e-11 1.75199326e-06 2.08404657e+00 4.25673560e-01]
[3.92780507e+00 3.79829049e+03 4.60646509e+00 1.85283244e+00]
1
  """

  # Запускаем мультистарт
  res, curve = solver.multistart(lb, ub, quantity_starts, ralgb5, return_curve_progress=True, return_all_data=False)
  print(*res, sep="\n")


  pass


