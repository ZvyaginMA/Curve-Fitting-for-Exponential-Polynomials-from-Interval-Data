import numpy as np

class Tol_with_cost:
    """
        Класс распознающего функционала со штрафной функцией
    """
    def __init__(self, x_lb: np.ndarray, x_ub: np.ndarray, y_lb: np.ndarray, y_ub: np.ndarray, cost_a: np.ndarray, cost_b: np.ndarray):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.cost_a = cost_a
        self.cost_b = cost_b

    def tol_value(self, a_coefficients: np.ndarray, b_coefficients: np.ndarray):
        y_rad = 0.5 * (self.y_ub - self.y_lb)
        y_mid = 0.5 * (self.y_ub + self.y_lb)
        a, b = a_coefficients, b_coefficients
        pf = self.cost_a @ (0.5 * (a_coefficients - np.abs(a_coefficients))) + self.cost_b @ (0.5 * (b_coefficients - np.abs(b_coefficients)))
        val = min([y_rad[i] - 0.5 * (np.exp(-self.x_lb[i] * b) @ (a) - np.exp(-self.x_ub[i] * b) @ (a)) -
                   np.abs(0.5 * (np.exp(-self.x_lb[i] * b) @ (a) + np.exp(-self.x_ub[i] * b) @ (a)) - y_mid[i]) for i in range(len(y_mid))])
        return val + pf

    def dTda(self, a: np.ndarray, b: np.ndarray):
        y_rad = 0.5 * (self.y_ub - self.y_lb)
        y_mid = 0.5 * (self.y_ub + self.y_lb)

        index_min = np.argmin([y_rad[i] - 0.5 * (np.exp(-self.x_lb[i] * b) @ (a) - np.exp(-self.x_ub[i] * b) @ (a)) - np.abs(
            0.5 * (np.exp(-self.x_lb[i] * b) @ (a) + np.exp(-self.x_ub[i] * b) @ (a)) - y_mid[i]) for i in range(len(y_mid))])
        e_lb = np.exp(-self.x_lb[index_min] * b)
        e_ub = np.exp(-self.x_ub[index_min] * b)
        #print(e_lb, e_ub)
        gpf = - self.cost_a * np.sign(0.5 * (a - np.abs(a)))
        grad = -0.5 * (e_lb - e_ub) - np.sign(0.5 * ((e_lb + e_ub) @ (a)) - y_mid[index_min]) * (0.5 * (e_lb + e_ub))
        return grad + gpf

    def dTdb(self, a: np.ndarray, b: np.ndarray):
        y_rad = 0.5 * (self.y_ub - self.y_lb)
        y_mid = 0.5 * (self.y_ub + self.y_lb)

        index_min = np.argmin([y_rad[i] - 0.5 * (np.exp(-self.x_lb[i] * b) @ (a) - np.exp(-self.x_ub[i] * b) @ (a)) - np.abs(
            0.5 * (np.exp(-self.x_lb[i] * b) @ (a) + np.exp(-self.x_ub[i] * b) @ (a)) - y_mid[i]) for i in range(len(y_mid))])
        e_lb = np.exp(-self.x_lb[index_min] * b)
        e_ub = np.exp(-self.x_ub[index_min] * b)
        gpf = - self.cost_b * np.sign(0.5 * (b - np.abs(b)))
        grad = 0.5 * (self.x_lb[index_min] * e_lb - self.x_ub[index_min] * e_ub) * a + np.sign(
            0.5 * ((e_lb + e_ub) @ (a)) - y_mid[index_min]) * 0.5 * (
                           self.x_lb[index_min] * e_lb + self.x_ub[index_min] * e_ub) * a
        return grad + gpf

