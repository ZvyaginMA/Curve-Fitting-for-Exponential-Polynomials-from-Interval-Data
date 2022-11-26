import unittest
import numpy as np

from tol_func import Tol

class TestSolver(unittest.TestCase):

    def test_tol_1(self):
        y_mid = np.array([2.185, 1.475, 1.2075, 0.98, 0.85, 0.7275])
        y_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.15
        y_lb = y_mid - y_rad
        y_ub = y_mid + y_rad
        x_mid = np.array([0.1, 2.0, 4.0, 6.0, 8.0, 10.0])
        x_rad = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.1
        x_lb = x_mid - x_rad
        x_ub = x_mid + x_rad
        # класс распознающего функционала
        tol = Tol(x_lb, x_ub, y_lb, y_ub)

        a = np.array([0.1, 1.1])
        b = np.array([0.1, 0.1])

        expected_val = -0.8587615920318936
        expected_dTda = np.array([0.98019867, 0.98019867])
        expected_dTdb = np.array([-0.01960397, - 0.21564371])

        actual_val = tol.tol_value(a, b)
        actual_dTda = tol.dTda(a, b)
        actual_dTdb = tol.dTdb(a, b)

        self.assertTrue(abs(expected_val - actual_val) < 1.e-7)
        self.assertTrue(np.linalg.norm(expected_dTda - actual_dTda) < 1.E-7)
        self.assertTrue(np.linalg.norm(expected_dTdb - actual_dTdb) < 1.E-7)
