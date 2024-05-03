import numpy as np
from curve_fitting.exp_model import ExpModel
from curve_fitting.simple_extruder import SimpleExrtuder
from curve_fitting.tol import Tol
import plot.plot_interval as pi
import pandas as pd
from curve_fitting.tol_lin_mixin import TolLinMixin
from curve_fitting.solver import Solver
from curve_fitting.optimization import ralgb5

class ex:

    @staticmethod
    def p1():
        """
            Пример со степенным слагаемым
        """ 
        x_m = np.array([0.12, 0.2 , 0.6 , 0.7 , 0.9 , 1.4 , 1.6 , 1.9 , 2.4 ])
        y_lb = np.array([ 2.564 ,  1.0253,  0.2068,  0.1517,  0.0753, -0.017 , -0.0343,-0.0501, -0.0623])
        y_ub = np.array([2.704 , 1.1653, 0.3468, 0.2917, 0.2153, 0.123 , 0.1057, 0.0899, 0.0777])

        f1 = lambda x: 0.01 * x**-2.5 + 0.8 * np.exp(-2 * x)
        quantity_exp = 1
        
        # класс распознающего функционала
        f = [lambda x: x**(-2.5)]
        tol = TolLinMixin(x_m , x_m , y_lb, y_ub, 2* np.ones(quantity_exp),3* np.ones(quantity_exp), 4*np.ones(1), f )
        # Передаём функционал в солвер
        solver = Solver(tol)

        #Задаём область из которой выбираются начальные приближения для мультистартов
        lb = 0 * np.ones(quantity_exp * 2 + 1)
        ub = 4 * np.ones(quantity_exp * 2 + 1)
        quantity_starts = 100

        # Запускаем мультистарт 
        # {'best_res': {'func_val': 0.06953377113648186, 'a': array([0.79718602]), 'b': array([1.99189043]), 'c': array([0.01001037]), 'code': 3}}
        res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)
        f_res = lambda x: 0.79718602 * np.exp(- 1.99189043 * x) + 0.01001037* x**(-2.5)

        pi.draw_interval_and_f([f1], x_m - 0.01, x_m + 0.01, y_lb, y_ub)
        pi.show()

        pass

    @staticmethod
    def p2():
        """
            Пример с изменением вогнутости
        """ 
        x_m = np.array([0.12, 0.2 , 0.6 , 0.7 , 0.9 , 1.4 , 1.6 , 1.9 , 2.4 ])
        y_lb = np.array([1.7147, 1.5584, 1.1539, 1.105 , 1.0196, 0.7212, 0.5615, 0.3262, 0.0575])
        y_ub = np.array([1.8547, 1.6984, 1.2939, 1.245 , 1.1596, 0.8612, 0.7015, 0.4662, 0.1975])

        f2 = lambda x: 0.8 * np.exp(-(x - 1)**2)  + 1.8 * np.exp(-2 * x)

        quantity_exp = 1
        
        # класс распознающего функционала
        f = [lambda x: np.exp(-(x - 1)**2)]
        tol = TolLinMixin(x_m , x_m , y_lb, y_ub, 2* np.ones(quantity_exp),3* np.ones(quantity_exp), 4*np.ones(1), f )
        # Передаём функционал в солвер
        solver = Solver(tol)

        #Задаём область из которой выбираются начальные приближения для мультистартов
        lb = 0 * np.ones(quantity_exp * 2 + 1)
        ub = 4 * np.ones(quantity_exp * 2 + 1)
        quantity_starts = 100

        # Запускаем мультистарт solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)
        #  {'best_res': {'func_val': 0.06896398598907982, 'a': array([1.80237325]), 'b': array([2.01527691]), 'c': array([0.8037996]), 'code': 3}}
        res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)
        f_res = lambda x: 0.80763673 * np.exp(- 1.99672163 * x) + 0.00992973* x**(-2.5)
        res
        pi.draw_interval_and_f([f2], x_m - 0.01, x_m + 0.01, y_lb, y_ub)
        pi.show()

        pass

    @staticmethod
    def p3():
        """
            Пример с изменением вогнутости
        """ 
        x_m = np.array([0.12, 0.2 , 0.35, 0.9 , 1.2 , 1.9 , 2.4 ])
        y_lb = np.array([3.8448, 4.1745, 4.4482, 4.0512, 3.5344, 2.3167, 1.6197])
        y_ub = np.array([3.9848, 4.3145, 4.5882, 4.1912, 3.6744, 2.4567, 1.7597])
 
        f3 = lambda x: 10 * x**0.7 *  np.exp(-x)  + 1.8 * np.exp(-2 * x) + 0.7 * np.exp(-3 * x)

        quantity_exp = 2
        
        # класс распознающего функционала
        f = [lambda x: x**0.7 * np.exp(-x)]
        tol = TolLinMixin(x_m , x_m , y_lb, y_ub, 2* np.ones(quantity_exp),3* np.ones(quantity_exp), 4*np.ones(1), f )
        # Передаём функционал в солвер
        solver = Solver(tol)

        #Задаём область из которой выбираются начальные приближения для мультистартов
        lb = 0 * np.ones(quantity_exp * 2 + 1)
        ub = 4 * np.ones(quantity_exp * 2 + 1)
        quantity_starts = 100

        # Запускаем мультистарт solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)
        #  {'best_res': {'func_val': 0.0695988939822183, 'a': array([2.00535398, 0.49208147]), 'b': array([2.06772089, 3.1452198 ]), 'c': array([10.00886183]), 'code': 3}}
        res = solver.multistart(lb, ub, quantity_starts, ralgb5, return_all_data=False)
        a = res["best_res"]["a"]
        b = res["best_res"]["b"]
        c = res["best_res"]["c"]
        f_res = lambda x: a @ np.exp(- b * x) + c[0]* f[0](x)
        a = res["best_res"]["a"]
        b = res["best_res"]["b"]
        c = res["best_res"]["c"]
        f_res = lambda x: a @ np.exp(- b * x) + c[0]* f[0](x)

        pi.draw_interval_and_f([f3], x_m - 0.01, x_m + 0.01, y_lb, y_ub)
        pi.show()

        pass

    @staticmethod
    def NegatationCoef():

        x_mid = np.array([0.2 , 0.35, 0.9 , 1.2 , 1.9 , 2.4 ]),
        y_lb = np.array([0.7442, 0.8145, 0.6423, 0.4914, 0.2241, 0.1103]),
        y_ub = np.array([0.8842, 0.9545, 0.7823, 0.6314, 0.3641, 0.2503])

        model = ExpModel(quantity_exp= 2)
        model.Fit(x_mid, x_mid, y_lb, y_ub, cost_a=0)
        pass
