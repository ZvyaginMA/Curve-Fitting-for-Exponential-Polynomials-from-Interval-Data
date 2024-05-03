from curve_fitting.tol import Tol
from curve_fitting.exp_model import ExpModel
import numpy as np
import plot.plot_interval as pi

class SimpleExrtuder:
    @staticmethod
    def Extrude( tol : Tol, a, b, a_direction, b_direction, step_a = 1.0, step_b = 1.0):
        tol.use_a_cost = False
        tol.use_b_cost = False

        if(tol.tol_value(a, b) < 0):
            raise Exception("Tol value must be greater zero in start point")

        tv = 10000
        new_a = a
        new_b = b
        while(tv > 0):
            new_a = new_a + step_a * a_direction
            new_b = new_b + step_b * b_direction
            tv = tol.tol_value(new_a, new_b)

        outer_a = new_a
        outer_b = new_b
        inner_a = new_a - step_a * a_direction
        inner_b = new_b - step_b * b_direction
        dist = 100
        while(dist> 0.01):
            new_a = 0.5 * (inner_a + outer_a)
            new_b = 0.5 * (inner_b + outer_b)
            tv = tol.tol_value(new_a, new_b)    
            if(tv > 0.0):
                inner_a = new_a
                inner_b = new_b
            else:
                outer_a = new_a
                outer_b = new_b
            n_a = (inner_a - outer_a) @ (inner_a - outer_a)     
            n_b = (inner_b - outer_b) @ (inner_b - outer_b)   
            dist = (n_a + n_b)**0.5 
        return inner_a, inner_b
        
    @staticmethod
    def ExtrudeNP(x_lb: np.ndarray, x_ub: np.ndarray, y_lb: np.ndarray, y_ub: np.ndarray, quantity_exp):
        model = ExpModel(quantity_exp= quantity_exp)
        a = []
        b = []
        model.Fit(x_lb, x_ub, y_lb, y_ub)
        a.append(model.a_coef)
        b.append(model.b_coef)
        x_lb_1 = 0.15 *x_lb +0.85 * x_ub
        y_lb_1 = 0.15 *y_lb + 0.85 *y_ub
        x_ub_1 = 0.85 *x_lb +0.15 * x_ub
        y_ub_1 = 0.85 *y_lb +0.15 * y_ub
        model.Fit(x_lb_1, x_ub, y_lb_1, y_ub)
        a.append(model.a_coef)
        b.append(model.b_coef)
        model.Fit(x_lb, x_ub_1, y_lb, y_ub_1)
        a.append(model.a_coef)
        b.append(model.b_coef)
        pi.draw_many(a,b, x_lb, x_ub, y_lb, y_ub)
        pi.draw_interval(x_lb, x_ub, y_lb, y_ub)
        pi.show()
        