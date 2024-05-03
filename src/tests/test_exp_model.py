import unittest
import numpy as np
from curve_fitting.exp_model import ExpModel, ExpLinModel
import pandas as pd
import matplotlib.pyplot as plt
import plot.plot_interval as pi

class TestExpModel(unittest.TestCase):
    def test_exp(self):
        y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93])
        y_rad = np.ones(6) * 0.005
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = np.array(range(0, 6, 1)) * 0.05
        x_rad = np.ones(6) * 0.0
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad

        model = ExpModel(quantity_exp= 4)
        model.Fit(x_lb, x_ub, y_lb, y_ub)
        tol_val = model.tol_value
        self.assertGreater(tol_val, 0.0015)

    def test_exp2(self):
        df = pd.read_excel("D:\Py\ExpCurvFit\Пример данных для ИВТ СО РАН.xlsx")
        t = np.array(df["Time(ms)"][20:35])
        y = np.array(df["EMF(mV)"][20:35])
        y_sigma = np.array(df["Rel. Error"][20:35])
        y_mid = y
        y_rad = y_sigma * 3
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = t
        print(y_sigma, y, t)
        x_rad = 0.01
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad
        f = [lambda x: x**(-2.5)]

        model = ExpLinModel(quantity_exp= 3,func_list= f)
        model.Fit(x_lb, x_ub, y_lb, y_ub, cost_a=10, cost_b=10, cost_c=10)
        print(model.a_coef, model.b_coef,model.c_coef, model.tol_value)

        #self.assertGreater(model.tol_value, 0.015)
        limit_lb = 0
        limit_ub = 15
        pi.draw_interval_and_f(model.Predict,x_lb[limit_lb:limit_ub], x_ub[limit_lb:limit_ub], y_lb[limit_lb:limit_ub], y_ub[limit_lb:limit_ub])
        pi.show()
        #plt.scatter(t[limit_lb:limit_ub], y_lb[limit_lb:limit_ub])
        #plt.scatter(t[limit_lb:limit_ub], y_ub[limit_lb:limit_ub])
        #plt.scatter(t[limit_lb:limit_ub], y[limit_lb:limit_ub])
        #x = np.arange(t[limit_lb], t[limit_ub], 1)
        #y_pred_local = np.array([model.Predict(tt) for tt in x])
        #plt.plot(x, y_pred_local, c = "r")
        #plt.show()
