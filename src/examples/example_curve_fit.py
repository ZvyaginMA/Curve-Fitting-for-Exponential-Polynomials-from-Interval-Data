import numpy as np
from curve_fitting.exp_model import ExpModel
from curve_fitting.simple_extruder import SimpleExrtuder
from curve_fitting.tol import Tol
import plot.plot_interval as pi
import pandas as pd

def ex1():
    # Задаем данные
    y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93])
    y_rad = np.ones(6) * 0.9
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(0, 6, 1)) * 1.0
    x_rad = np.ones(6) * 0.3
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad

    # Создаем модель и задаем количество экспонент
    model = ExpModel(quantity_exp= 2)
    # Расчет
    model.Fit(x_lb, x_ub, y_lb, y_ub)
    # Выводим результат
    print(model.tol_value , model.a_coef , model.b_coef, model.min_rad_y)
    pi.draw_interval_and_f([model.Predict], x_lb, x_ub, y_lb, y_ub)
    pi.show()
    pass


def ex2():
    y_mid = np.array([1.9 , 2.04, 1.67, 1.37, 1.12, 0.93])
    y_rad = np.ones(6) * 0.3
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(0, 6, 1)) * 1.0
    x_rad = np.ones(6) * 0.0
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    """a_max = np.array([1.8263335,  0.68420739])
    b_max = np.array([0.24085562, 0.11819026])"""

    model = ExpModel(quantity_exp= 2)
    model.Fit(x_lb, x_ub, y_lb, y_ub, cost_a=0)
    a_max = model.a_coef
    b_max = model.b_coef
    tol = Tol(x_lb, x_ub, y_lb, y_ub)
    a1, b1 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([1.0, 0]), np.ones((2)))
    a2, b2 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([0, 1.0]), np.ones((2)))
    a3, b3 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.ones((2)), np.array([1.0, 0]))
    a4, b4 = SimpleExrtuder.Extrude(tol, a_max, b_max ,  np.ones((2)), np.array([1.0, 0]))
    a5, b5 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([1.0, 1.0]), np.ones((2)))
    a6, b6 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.ones((2)), np.array([1.0, 1.0]))
    a7, b7 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([1.0,0]), np.array([0, 1.0]))
    a8, b8 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([0,1.0]), np.array([0,1.0]))

    pi.draw_many([a_max, a1, a2, a3, a4, a5,a6, a7, a8],[b_max, b1, b2, b3, b4, b5,b6,b7,b8], x_lb, x_ub, y_lb, y_ub)
    a1, b1 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([-1.0, 0]), np.ones((2)))
    a2, b2 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([0, -1.0]), np.ones((2)))
    a3, b3 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.ones((2)), np.array([-1.0, 0]))
    a4, b4 = SimpleExrtuder.Extrude(tol, a_max, b_max ,  np.ones((2)), np.array([-1.0, 0]))
    a5, b5 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([-1.0, -1.0]), np.ones((2)))
    a6, b6 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.ones((2)), np.array([-1.0, -1.0]))
    a7, b7 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([1.0,0]), np.array([0, 1.0]))
    a8, b8 = SimpleExrtuder.Extrude(tol, a_max, b_max , np.array([0,1.0]), np.array([0,1.0]))

    pi.draw_many([a_max, a1, a2, a3, a4, a5,a6, a7, a8],[b_max, b1, b2, b3, b4, b5,b6,b7,b8], x_lb, x_ub, y_lb, y_ub)
    pi.draw_interval(x_lb, x_ub, y_lb, y_ub)
    pi.show()
    print(tol.tol_value(a1, b1))
    print(tol.tol_value(a2, b2))
    print(tol.tol_value(a3, b3))
    print(tol.tol_value(a4, b4))


def ex4():
    y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93])
    y_rad = np.ones(6) * 0.3
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(0, 6, 1)) * 1.0
    x_rad = np.ones(6) * 0.0
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    SimpleExrtuder.ExtrudeNP(x_lb, x_ub, y_lb, y_ub, 2)

def ex5():
    y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93])
    y_rad = np.array([0.1, 0.3, 0.2, 0.2, 0.1, 0.3])
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(0, 6, 1)) * 1.0
    x_rad = np.ones(6) * 0.3
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    SimpleExrtuder.ExtrudeNP(x_lb, x_ub, y_lb, y_ub, 2)

def ReadData():
    df = pd.read_excel("D:\Py\ExpCurvFit\examples\Пример данных для ИВТ СО РАН.xlsx")
    pass
    print(df.describe())
    x = df["Time(ms)"]
    y = df["EMF(mV)"]
    y_sigma = df["Rel. Error"]
    
    y_mid = np.array(y)
    y_rad = np.array(y_sigma) * 2
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(x)
    x_rad = 0
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad
    model = ExpModel(quantity_exp= 4)
    model.Fit(x_lb, x_ub, y_lb, y_ub)
    print(model.tol_value , model.a_coef , model.b_coef, model.min_rad_y)
    pass

def LancEx1():
    y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93, 0.77, 0.64, 0.53, 0.45, 0.38,0.32, 0.27, 0.23, 0.2, 0.17, 0.15, 0.13, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06])
    y_rad = np.ones(24) * 0.005
    y_lb = y_mid - y_rad
    y_ub = y_mid + y_rad
    x_mid = np.array(range(0, 24, 1)) * 0.05
    x_rad = np.ones_like(x_mid) * 0.0
    x_lb = x_mid - x_rad
    x_ub = x_mid + x_rad

    model = ExpModel(quantity_exp= 2)
    model.Fit(x_lb, x_ub, y_lb, y_ub)
    print(model.tol_value , model.a_coef , model.b_coef, model.min_rad_y)
    pi.draw_interval(x_lb, x_ub, y_lb, y_ub)
    pi.show()
    pi.draw_interval(x_lb, x_ub, y_lb, y_ub)
    pi.show()
    pi.draw_interval_and_f(model.Predict, x_lb, x_ub, y_lb, y_ub)
    pi.show()