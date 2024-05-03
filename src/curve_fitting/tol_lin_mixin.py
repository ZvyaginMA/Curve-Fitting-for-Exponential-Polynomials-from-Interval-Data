import numpy as np

class TolLinMixin:
    """
        Класс распознающего функционала со штрафной функцией
    """
    def __init__(self, x_lb: np.ndarray, 
                 x_ub: np.ndarray, 
                 y_lb: np.ndarray, 
                 y_ub: np.ndarray, 
                 cost_a: np.ndarray = None, 
                 cost_b: np.ndarray = None,
                 cost_c: np.ndarray = None,
                 func_mixin = None,
                 weight : np.ndarray = None):
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.y_lb = y_lb
        self.y_ub = y_ub
        self.use_a_cost = cost_a is not None
        self.use_b_cost = cost_b is not None
        self.use_c_cost = cost_c is not None
        self.use_weight = weight is not None
        self.cost_a = cost_a
        self.cost_b = cost_b
        self.cost_c = cost_c
        self.f = func_mixin
        self.dim_mixin = len(func_mixin)
        if(self.use_weight):
            self.weight = weight
        else:
            self.weight = np.ones_like(x_lb)
        if self.use_c_cost and (len(func_mixin) != len(cost_c)):
            raise Exception("Dimension of vector cost_c and func_mixin must coincide")
        else:
            self.dim_mixin = len(cost_c)
        

    def tol_value(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        if(self.use_a_cost):
            pf_a =  self.cost_a @ (0.5 * (a - np.abs(a)))
        else:
            pf_a = 0
        if(self.use_b_cost):
            pf_b = self.cost_b @ (0.5 * (b - np.abs(b)))
        else:
            pf_b = 0
        if(self.use_c_cost):
            pf_c = self.cost_c @ (0.5 * (c - np.abs(c)))
        else:
            pf_c = 0

        val = min(self.__calc_generatrix(a, b, c))
        return val + pf_a + pf_b + pf_c
    
    def __calc_generatrix(self, a_coef: np.ndarray, b_coef: np.ndarray, c_coeff: np.ndarray):
        y_rad = 0.5 * (self.y_ub - self.y_lb)
        y_mid = 0.5 * (self.y_ub + self.y_lb)

        return self.weight * np.array([y_rad[i] - 0.5 * (np.exp(-self.x_lb[i] * b_coef) @ (a_coef) - np.exp(-self.x_ub[i] * b_coef) @ (a_coef)) 
                        - 0.5 * c_coeff @ np.array([np.sign(ff(self.x_ub[i])-ff(self.x_lb[i])) * (ff(self.x_ub[i])-ff(self.x_lb[i])) for ff in self.f])
                        - np.abs(0.5 * (np.exp(-self.x_lb[i] * b_coef) @ (a_coef) + np.exp(-self.x_ub[i] * b_coef) @ (a_coef)) - y_mid[i]  
                        + 0.5 * c_coeff @ np.array([(ff(self.x_ub[i])+ff(self.x_lb[i])) for ff in self.f])) for i in range(len(y_mid))])

    def dTda(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        y_mid = 0.5 * (self.y_ub + self.y_lb)
        index_min = np.argmin(self.__calc_generatrix(a, b, c))
        e_lb = np.exp(-self.x_lb[index_min] * b)
        e_ub = np.exp(-self.x_ub[index_min] * b)
        if(self.use_a_cost):
            gpf =  - self.cost_a * np.sign(0.5 * (a - np.abs(a)))
        else:
            gpf = 0

        f_lb = np.array([ff(self.x_lb[index_min]) for ff in self.f])
        f_ub = np.array([ff(self.x_ub[index_min]) for ff in self.f])
        Phi = np.sign(y_mid[index_min] - 0.5 * ((e_lb + e_ub) @ (a)) - 0.5 * c @ (f_ub + f_lb))

        grad = (-0.5 * (e_lb - e_ub) + Phi * (0.5 * (e_lb + e_ub)))
        return self.weight[index_min] * grad + gpf

    def dTdb(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        y_mid = 0.5 * (self.y_ub + self.y_lb)
        index_min = np.argmin(self.__calc_generatrix(a, b, c))
        e_lb = np.exp(-self.x_lb[index_min] * b)
        e_ub = np.exp(-self.x_ub[index_min] * b)

        if(self.use_b_cost):
            gpf =  - self.cost_b * np.sign(0.5 * (b - np.abs(b)))
        else:
            gpf = 0
        
        f_lb = np.array([ff(self.x_lb[index_min]) for ff in self.f])
        f_ub = np.array([ff(self.x_ub[index_min]) for ff in self.f])
        Phi = np.sign(y_mid[index_min] - 0.5 * ((e_lb + e_ub) @ (a)) - 0.5 * c @ (f_ub + f_lb))

        grad = 0.5 * (self.x_lb[index_min] * e_lb - self.x_ub[index_min] * e_ub) * a 
        - Phi * 0.5 * (self.x_lb[index_min] * e_lb + self.x_ub[index_min] * e_ub) * a
        return self.weight[index_min] * grad + gpf
    
    def dTdc(self, a: np.ndarray, b: np.ndarray, c: np.ndarray):
        y_mid = 0.5 * (self.y_ub + self.y_lb)
        index_min = np.argmin(self.__calc_generatrix(a, b, c))
        e_lb = np.exp(-self.x_lb[index_min] * b)
        e_ub = np.exp(-self.x_ub[index_min] * b)

        if(self.use_c_cost):
            gpf =  - self.cost_c * np.sign(0.5 * (c - np.abs(c)))
        else:
            gpf = 0

        f_lb = np.array([ff(self.x_lb[index_min]) for ff in self.f])
        f_ub = np.array([ff(self.x_ub[index_min]) for ff in self.f])
        Phi = np.sign(y_mid[index_min] - 0.5 * ((e_lb + e_ub) @ (a)) - 0.5 * c @ (f_ub + f_lb))
        grad = 0.5 * np.sign(f_ub - f_lb)*(f_ub - f_lb) + 0.5 * Phi * (f_ub + f_lb)
        return self.weight[index_min] * grad + gpf

    def get_calcfg(self):
        def calcfg(x):
            a, b, c = x[:(len(x) - self.dim_mixin) // 2], x[(len(x) - self.dim_mixin) // 2:-self.dim_mixin], x[-self.dim_mixin:]
            return - self.tol_value(a, b, c), np.concatenate(
            [- self.dTda(a, b, c), - self.dTdb(a, b, c), - self.dTdc(a, b, c)])
        return calcfg
        
    def optimization(self,method , x_0):
        def calcfg(x):
            a, b, c = x[:(len(x) - self.dim_mixin) // 2], x[(len(x) - self.dim_mixin) // 2:-self.dim_mixin], x[-self.dim_mixin:]
            return - self.tol_value(a, b, c), np.concatenate(
            [- self.dTda(a, b, c), - self.dTdb(a, b, c), - self.dTdc(a, b, c)])
        xr, fr, nit, ncalls, ccode = method(calcfg, x_0, tolg =1e-16) 
        a, b, c = xr[:(len(xr) - self.dim_mixin) // 2], xr[(len(xr) - self.dim_mixin) // 2:-self.dim_mixin], xr[-self.dim_mixin:]
        return {"func_val": - fr,"a": a,"b": b, "c": c,"code": ccode }