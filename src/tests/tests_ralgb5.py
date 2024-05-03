import unittest
import numpy as np
from curve_fitting.optimization import ralgb5
from curve_fitting.tol import Tol

class TestRalgb(unittest.TestCase):
    def test_ralgb_on_calcfg1(self):
        x_start = np.array([4.0, 2.0])
        def calcfg1(x):
            """
                f(x) = x @ x
                g(x) = 2 * x
            """
            return x @ x , 2 * x
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg1, x_start)
        self.assertTrue(abs(xr).sum() < 1E-5)
        self.assertTrue(abs(fr).sum() < 1E-10)

    def test_ralgb_on_calcfg2(self):
        x_start = np.array([4.0])
        def calcfg2(x):
            """
                f(x) = x^2 * cos(14x) + 1.7x^2
                g(x) = 2x * cos(14x) - 14 *x^2 * sin(14x) + 3.4x
            """
            if (len(x) > 1):
                raise Exception(f"x must have len = 1 but have {len(x)}")
            f = x*x * np.cos(4 * x) + 1.7 * x * x
            g = 2 * x * np.cos(14 * x) - 14 *x * x * np.sin(14 * x) + 3.4 * x
            return f, g
        
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg2, x_start)
        self.assertTrue(abs(xr - 2.37826).sum() < 1E-4)
        # нашли локальный минимум

    def test_ralgb_with_nan_err(self):
        y_mid = np.array([2.51, 2.04, 1.67, 1.37, 1.12, 0.93])
        y_rad = np.array([0.1, 0.3, 0.2, 0.2, 0.1, 0.3])
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = np.array(range(0, 6, 1)) * 1.0
        x_rad = np.ones(6) * 0.3
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad

        cost_a = 10.0 *np.ones(2)
        cost_b = 10.0 *np.ones(2)
        tol1 = Tol(x_lb, x_ub, y_lb, y_ub, cost_a , cost_b)

        def calcfg(x):
            a, b = x[:len(x) // 2], x[len(x) // 2:]
            return - tol1.tol_value(a, b), np.concatenate(
                [- tol1.dTda(a, b), - tol1.dTdb(a, b)])

        res = ralgb5(calcfg, np.array([10.0, -11.0, -14.0, 18.0]))
        self.assertLess(np.abs(res[1] - -0.0276), 1e-3)
        
        