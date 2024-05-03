import unittest
import numpy as np
from curve_fitting.tol_lin_mixin import TolLinMixin
from curve_fitting.tol import Tol
class TestMixin(unittest.TestCase):
    def test_val(self):
        # Arrange
        y_mid = np.array([2.0])
        y_rad = np.ones(1)
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = np.array([1]) * 1.0
        x_rad = np.ones(1) * 0.1
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad
        f = [lambda x: x, lambda x: x**(-2.5)]
        tol = TolLinMixin(x_lb, x_ub, y_lb, y_ub, 2* np.ones(2),3* np.ones(2), 4*np.ones(2), f )
        expected = 1 - (2 * 0.5 * 0.2 - 1* 0.5 *(1.1**(-2.5) - 0.9**(-2.5))) - np.abs(2 - 0.5 *(2 * (1.1+0.9) + 1 * (1.1**(-2.5) + 0.9**(-2.5))))

        # Act
        res = tol.tol_value(np.array([0, 0]), np.array([0, 0]), np.array([2, 1]))

        # Assert
        self.assertTrue(np.abs(expected -res) < 1e-8)

    def test_val2(self):
        # Arrange
        y_mid = np.array([2.0])
        y_rad = np.ones(1)
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = np.array([1]) * 1.0
        x_rad = np.ones(1) * 0.1
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad
        f = [lambda x: x, lambda x: x**(-2.5)]
        tol = Tol(x_lb, x_ub, y_lb, y_ub, 2* np.ones(2),3* np.ones(2))
        tol_lin = TolLinMixin(x_lb, x_ub, y_lb, y_ub, 2* np.ones(2),3* np.ones(2), 4*np.ones(2), f )
        # Act
        res_ex = tol.tol_value(np.array([1, 1]), np.array([1, 1]))
        res_lin = tol_lin.tol_value(np.array([1, 1]), np.array([1, 1]), np.array([0, 0]))
        # Assert
        self.assertTrue(np.abs(res_ex -res_lin) < 1e-8)

    def test_dTdc(self):
        # Arrange
        y_mid = np.array([2.0])
        y_rad = np.ones(1)
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = np.array([1]) * 1.0
        x_rad = np.ones(1) * 0.1
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad
        f = [lambda x: x, lambda x: x**(-2.5)]
        tol = TolLinMixin(x_lb, x_ub, y_lb, y_ub, 2* np.ones(2),3* np.ones(2), 4*np.ones(2), f )
        ex1 = 0.5 * 0.2 + 0.5*np.sign(2 - 2 - 0.5 *(1.1**(-2.5) + (0.9)**(-2.5))) *2  
        ex2 = -0.5 * (1.1**(-2.5) - (0.9)**(-2.5)) + 0.5*np.sign(2 - 2 - 0.5 *(1.1**(-2.5) + (0.9)**(-2.5))) *(1.1**(-2.5) + (0.9)**(-2.5))
        expected_grad = np.array([ex1, ex2])
        # Act
        grad = tol.dTdc(np.array([0, 0]), np.array([0, 0]), np.array([2.0, 1]))

        # Assert
        self.assertTrue(np.max(np.abs(grad - expected_grad)) < 1e-8)