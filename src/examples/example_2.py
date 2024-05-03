import numpy as np
from curve_fitting.tol_lin_mixin import TolLinMixin
import plot.plot_interval as pi
import matplotlib.pyplot as plt
from curve_fitting.solver import Solver
from curve_fitting.optimization import ralgb5
import examples.example_mixin 
#example_curve_fit.ex4()
def TR_data():
    y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93])
    y_rad = np.ones(6) * 0.9
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(0, 6, 1)) * 1.0
    x_rad = np.ones(6) * 0.3
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    pi.draw_interval(x_lb, x_ub, y_lb, y_ub)
    """
    В линеном случае Tol = 0.1012
    -0.316
    2.406
    TR = 0.34

    0.15354889905495628 [1.59315325 0.91257061] [0.19471132 0.19471135] 0.2999999999999998
    """
    pass

    y_mid = np.array([1, 2, 3, 4.0])
    y_rad = np.ones(4) * 0.3
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(1, 5, 1)) 
    x_rad = np.ones(4) * 0.3
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    pi.draw_interval(x_lb, x_ub, y_lb, y_ub)
    """
    array([ 1.00000000e+00, -1.07804574e-15]),
    mpf('-1.07804574452064039960388583566023382518e-15')
    """
    pass

    y_mid = np.array([1, 1.5, 2, 2.5])
    y_rad = np.ones(4) * 0.5
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(1, 5, 1)) 
    x_rad = np.ones(4) * 0.2
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    """
     (array([0.5, 0.5]), mpf('0.399999999999999344968415471157641150001')
    """
    pass

def t1():
    """
        расчет для сравнения линейной и нелиной функции
    """
    y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93])
    y_rad = np.ones(6) * 0.3
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(0, 6, 1)) * 1.0
    x_rad = np.ones(6) * 0.3
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    """
    В линеном случае Tol = 0.1012
    -0.316
    2.406
    TR = 0.34

    0.15354889905495628 [1.59315325 0.91257061] [0.19471132 0.19471135] 0.2999999999999998 0.512
    """
    a = np.array([1.59315325, 0.91257061])
    b = np.array([0.19471132, 0.19471135])
    pi.draw_interval_and_many_f([lambda t: -0.316 * t + 2.406, lambda x: a @ np.exp(-x * b)], x_lb, x_ub, y_lb, y_ub)
    pi.show()
    pass


def t2():
    """
        расчет для сравнения линейной и нелиной функции
    """
    y_mid = np.array([1, 2, 3, 4.0])
    y_rad = np.ones(4) * 0.3
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(1, 5, 1)) 
    x_rad = np.ones(4) * 0.3
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    pi.draw_interval(x_lb, x_ub, y_lb, y_ub)


    pi.draw_interval_and_many_f([lambda t: 1.0 * t + 0.0], x_lb, x_ub, y_lb, y_ub)
    pi.show()
    pass

def t3():
    """
    Пример зависимости TR от радиусов X

    rr = array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
       1.3, 1.4])
    tol = array([0.31666667, 0.29117647, 0.26666667, 0.24308176, 0.22037037,
       0.19848485, 0.17738095, 0.15701754, 0.13735632, 0.11836158,
       0.1       , 0.08397436, 0.06710526, 0.04932432, 0.02604167])

    y_mid = np.array([1.2, 1.6, 1.7, 2.5])
    y_rad = np.ones(4) * 0.5
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(1, 9, 2))
    x_rad = np.array([2.0, 0.4, 1.2, 0.8]) * rx
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
       0.2167
       0.8212
       """
    rr= np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2,
       1.3, 1.4])
    tol_list = np.array([0.31666667, 0.29117647, 0.26666667, 0.24308176, 0.22037037,
       0.19848485, 0.17738095, 0.15701754, 0.13735632, 0.11836158,
       0.1       , 0.08397436, 0.06710526, 0.04932432, 0.02604167])
    y_mid = np.array([1.2, 1.6, 1.7, 2.5])
    y_rad = np.ones(4) * 0.5
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(1, 9, 2))
    x_rad = np.array([2.0, 0.4, 1.2, 0.8]) * 0.1
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    pi.draw_interval(x_lb, x_ub, y_lb, y_ub)
    pi.show()
    
    
    plt.plot(rr, np.array(tol_list)/0.5, marker="o")    
    
    pass

t3()