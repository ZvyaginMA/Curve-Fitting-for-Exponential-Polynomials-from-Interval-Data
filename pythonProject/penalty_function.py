import numpy as np


class Penalty1():

    def Val(self, a_coefficients: np.ndarray, b_coefficients: np.ndarray):
        pf = self.cost_a @ (0.5 * (a_coefficients - np.abs(a_coefficients))) + self.cost_b @ (
                0.5 * (b_coefficients - np.abs(b_coefficients)))