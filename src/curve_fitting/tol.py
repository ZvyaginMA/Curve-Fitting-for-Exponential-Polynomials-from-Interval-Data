import numpy as np

class Tol:
    """
        Класс распознающего функционала со штрафной функцией
    """
    def __init__(self, x_lb: np.ndarray, 
                 x_ub: np.ndarray, 
                 y_lb: np.ndarray, 
                 y_ub: np.ndarray, 
                 cost_a: np.ndarray = None, 
                 cost_b: np.ndarray = None,
                 weight : np.ndarray = None):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.use_a_cost = cost_a is not None
        self.use_b_cost = cost_b is not None
        self.use_weight = weight is not None
        self.cost_a = cost_a
        self.cost_b = cost_b
        if(self.use_weight):
            self.weight = weight
        else:
            self.weight = np.ones_like(x_lb)

    def tol_value(self, a_coefficients: np.ndarray, b_coefficients: np.ndarray):
        a, b = a_coefficients, b_coefficients
        if(self.use_a_cost):
            pf_a =  self.cost_a @ (0.5 * (a_coefficients - np.abs(a_coefficients)))
        else:
            pf_a = 0
        if(self.use_b_cost):
            pf_b = self.cost_b @ (0.5 * (b_coefficients - np.abs(b_coefficients)))
        else:
            pf_b = 0

        val = min(self.__calc_generatrix(a, b))
        return val + pf_a + pf_b
    
    def __calc_generatrix(self, a_coef: np.ndarray, b_coef: np.ndarray):
        y_rad = 0.5 * (self.y_ub - self.y_lb)
        y_mid = 0.5 * (self.y_ub + self.y_lb)
        aa = np.exp(-self.x_lb[0] * b_coef)
        aa
        return self.weight * np.array([y_rad[i] - 0.5 * (np.exp(-self.x_lb[i] * b_coef) @ (a_coef) - np.exp(-self.x_ub[i] * b_coef) @ (a_coef)) -
                   np.abs(0.5 * (np.exp(-self.x_lb[i] * b_coef) @ (a_coef) + np.exp(-self.x_ub[i] * b_coef) @ (a_coef)) - y_mid[i]) for i in range(len(y_mid))])

    def dTda(self, a: np.ndarray, b: np.ndarray):
        y_mid = 0.5 * (self.y_ub + self.y_lb)
        index_min = np.argmin(self.__calc_generatrix(a, b))
        e_lb = np.exp(-self.x_lb[index_min] * b)
        e_ub = np.exp(-self.x_ub[index_min] * b)
        if(self.use_a_cost):
            gpf =  - self.cost_a * np.sign(0.5 * (a - np.abs(a)))
        else:
            gpf = 0
      
        grad = (-0.5 * (e_lb - e_ub) - np.sign(0.5 * ((e_lb + e_ub) @ (a)) - y_mid[index_min]) * (0.5 * (e_lb + e_ub)))
        return self.weight[index_min] * grad + gpf

    def dTdb(self, a: np.ndarray, b: np.ndarray):
        y_mid = 0.5 * (self.y_ub + self.y_lb)
        index_min = np.argmin(self.__calc_generatrix(a, b))
        e_lb = np.exp(-self.x_lb[index_min] * b)
        e_ub = np.exp(-self.x_ub[index_min] * b)

        if(self.use_b_cost):
            gpf =  - self.cost_b * np.sign(0.5 * (b - np.abs(b)))
        else:
            gpf = 0
        
        grad = 0.5 * (self.x_lb[index_min] * e_lb - self.x_ub[index_min] * e_ub) * a + np.sign(
            0.5 * ((e_lb + e_ub) @ (a)) - y_mid[index_min]) * 0.5 * (
                           self.x_lb[index_min] * e_lb + self.x_ub[index_min] * e_ub) * a
        return self.weight[index_min] * grad + gpf
    
    def get_calcfg(self):
        def calcfg(x):
            a, b = x[:len(x) // 2], x[len(x) // 2:]
            return - self.tol_value(a, b), np.concatenate(
            [- self.dTda(a, b), - self.dTdb(a, b)])
        return calcfg
    
    def optimization(self, method , x_0):
        def calcfg(x):
            a, b = x[:len(x) // 2], x[len(x) // 2:]
            return - self.tol_value(a, b), np.concatenate(
            [- self.dTda(a, b), - self.dTdb(a, b)])
        xr, fr, nit, ncalls, ccode = method(calcfg, x_0, tolg =1e-16) 
        a, b = xr[:len(xr) // 2], xr[len(xr) // 2:]
        
        return {"func_val": - fr,"a": a,"b": b, "code": ccode }
        
