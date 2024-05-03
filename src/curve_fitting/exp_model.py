import numpy as np
from curve_fitting.tol import Tol
from curve_fitting.tol_lin_mixin import TolLinMixin
from curve_fitting.solver import Solver
from curve_fitting.optimization import ralgb5

class ExpModel:
    def __init__(self, quantity_exp):
        self.quantity_exp = quantity_exp
        self.a_coef = None
        self.b_coef = None
        self.tol_value = None
        self.min_rad_y = None
        self.code = None

    def Fit(self, x_lb: np.ndarray, x_ub: np.ndarray, y_lb: np.ndarray, y_ub: np.ndarray, 
            cost_a : float = 10.0, cost_b : float  = 10.0, weight: np.ndarray = None):
        
        cost_a, cost_b = self.__costs(cost_a, cost_b)

        tol1 = Tol(x_lb, x_ub, y_lb, y_ub, cost_a , cost_b, weight)

        # Передаём функционал в солвер
        solver = Solver(tol1)

        #Задаём область из которой выбираются начальные приближения для мультистартов
        lb = 0 * np.ones(self.quantity_exp * 2)
        ub = 1 * np.ones(self.quantity_exp * 2)
        quantity_starts = 10

        # Запускаем мультистарт
        res = solver.multistart(lb, ub, quantity_starts, ralgb5,return_all_data=False)["best_res"]
        self.a_coef = res["a"]
        self.b_coef = res["b"]
        self.tol_value = res["func_val"] 
        self.code = res["code"] 
        #print(self.code)
        self.min_rad_y = 0.5 * np.min(y_ub - y_lb)

    def __costs(self, cost_a, cost_b):
        if(cost_a is None):
            pass
        else:
            c_a = cost_a *np.ones(self.quantity_exp)

        if(cost_b is None):
            pass
        else:
            c_b = cost_b *np.ones(self.quantity_exp)
        return c_a, c_b
        
    def Predict(self, t):
        return np.exp(-t * self.b_coef ) @ (self.a_coef)
    
    def SetA(self, a):
        self.a_coef = a

    def SetB(self, b):
        self.b_coef = b
        
class ExpLinModel(ExpModel):
    def __init__(self, quantity_exp, func_list):
        super().__init__(quantity_exp)
        self.c_coef = None
        self.func = func_list

    def Fit(self, x_lb: np.ndarray, x_ub: np.ndarray, y_lb: np.ndarray, y_ub: np.ndarray, 
            cost_a : float = 10.0, cost_b : float  = 10.0, cost_c : float = 10.0, weight: np.ndarray = None):
        
        cost_a, cost_b, cost_c = self.__cost(cost_a, cost_b, cost_c)

        tol1 = TolLinMixin(x_lb, x_ub, y_lb, y_ub, cost_a , cost_b, cost_c, self.func, weight)

        # Передаём функционал в солвер
        solver = Solver(tol1)

        #Задаём область из которой выбираются начальные приближения для мультистартов
        lb = 0 * np.ones(self.quantity_exp * 2 + len(self.func))
        ub = 1 * np.ones(self.quantity_exp * 2 + len(self.func))
        quantity_starts = 10

        # Запускаем мультистарт
        res = solver.multistart(lb, ub, quantity_starts, ralgb5,return_all_data=False)["best_res"]
        self.a_coef = res["a"]
        self.b_coef = res["b"]
        self.c_coef = res["c"]
        self.tol_value = res["func_val"] 
        self.code = res["code"] 
        #print(self.code)
        self.min_rad_y = 0.5 * np.min(y_ub - y_lb)

    def Predict(self, t):
        return np.exp(-t * self.b_coef ) @ (self.a_coef) + self.c_coef @ np.array([ff(t) for ff in self.func])
    
    def SetA(self, a):
        self.a_coef = a

    def SetB(self, b):
        self.b_coef = b

    def __cost(self, cost_a, cost_b, cost_c):
        if(cost_a is None):
            pass
        else:
            c_a = cost_a *np.ones(self.quantity_exp)

        if(cost_b is None):
            pass
        else:
            c_b = cost_b *np.ones(self.quantity_exp)

        if(cost_c is None):
            pass
        else:
            c_c = cost_c *np.ones(len(self.func))
        return c_a, c_b, c_c