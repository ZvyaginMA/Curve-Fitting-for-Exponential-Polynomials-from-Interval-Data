import numpy as np
from tol_func import Tol
from tol_func_mod import Tol_with_cost
from solver import Solve
from optimization_methods import ralgb5_with_proj, ralgb5
import unittest

class TestSolver(unittest.TestCase):

    def test_multistart_1(self):
        # Расчет мультистарта
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
        quantity_starts = 10

        # Запускаем мультистарт
        res = solver.multistart(lb, ub, quantity_starts, ralgb5_with_proj ,return_all_data=False)

        self.assertTrue(res[0] > 0.09) #Проверка ходимости
        self.assertTrue(res[-1] == 3) # Проверка кода остановки

    def test_multistart_2(self):
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
      cost_a = 5.0 *np.ones(quantity_exp)
      # стоимость выхода из области определения для b
      cost_b = 5.0 *np.ones(quantity_exp)
      # класс распознающего функционала
      tol1 = Tol_with_cost(x_lb, x_ub, y_lb, y_ub, cost_a , cost_b)

      # Передаём функционал в солвер
      solver = Solve(tol1)

      #Задаём область из которой выбираются начальные приближения для мультистартов
      lb = 0 * np.ones(quantity_exp * 2)
      ub = 1 * np.ones(quantity_exp * 2)
      quantity_starts = 10

      # Запускаем мультистарт
      res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_curve_progress= False,return_all_data=False)

      self.assertTrue(res[0] > 0.09)  # Проверка ходимости


