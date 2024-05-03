import unittest
import numpy as np
from curve_fitting.tol import Tol
from curve_fitting.tol_lin_mixin import TolLinMixin
from curve_fitting.solver import Solver
from curve_fitting.optimization import ralgb5

class TestSolver(unittest.TestCase):
    def test_multistart_1(self):
        """
        В этом примере тестируется решение оптимизационной задачи методом штрафной функции.
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
        tol1 = Tol(x_lb, x_ub, y_lb, y_ub, cost_a , cost_b)

        # Передаём функционал в солвер
        solver = Solver(tol1)

        #Задаём область из которой выбираются начальные приближения для мультистартов
        lb = 0 * np.ones(quantity_exp * 2)
        ub = 1 * np.ones(quantity_exp * 2)
        quantity_starts = 10

        # Запускаем мультистарт
        res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)

        self.assertTrue(res["best_res"]["func_val"] > 0.09)  # Проверка ходимости


    def test_multistart_2(self):
        
        y_mid = np.array([2.185, 1.475, 1.2075])
        y_rad = np.array([1.0, 1.0, 1.0]) * 0.15
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = np.array([0.14, 2.0, 4.0])
        x_rad = np.array([1.0, 1.0, 1.0]) * 0.1
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad


        quantity_exp = 1
        
        # класс распознающего функционала
        f = [lambda x: x**(-0.5)]
        tol = TolLinMixin(x_lb, x_ub, y_lb, y_ub, 10* np.ones(quantity_exp),10* np.ones(quantity_exp), 20*np.ones(1), f )
        # Передаём функционал в солвер
        solver = Solver(tol)

        #Задаём область из которой выбираются начальные приближения для мультистартов
        lb = 0 * np.ones(quantity_exp * 2 + 1)
        ub = 10 * np.ones(quantity_exp * 2 + 1)
        quantity_starts = 500

        # Запускаем мультистарт
        res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)

        self.assertTrue(res["best_res"]["func_val"] > 0.01)  # Проверка ходимости
        self.assertTrue(res["best_res"]["func_val"] < 0.07)