import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np

def draw_interval(f, a, b, x_lb : np.ndarray, x_ub : np.ndarray, y_lb : np.ndarray, y_ub : np.ndarray):
  plt.xlim(min(x_lb) - 0.5, max(x_ub) + 0.5)
  plt.ylim(min(y_lb) - 0.5, max(y_ub) + 0.5)
  axes = plt.gca()
  axes.set_aspect("equal")
  for i in range(len(x_lb)):
    drawRect(axes, x_lb[i], x_ub[i], y_lb[i], y_ub[i])
  x = np.arange(min(x_lb) - 0.5, max(x_ub) + 0.5, 0.1)
  fxx = np.array([f(a, b, i) for i in x])
  axes.plot(x, fxx)
  #plt.show()



def drawRect(axes, x_inf, x_sup, y_inf, y_sup):

    rect_coord = (x_inf, y_inf)
    rect_width = x_sup - x_inf
    rect_height = y_sup - y_inf
    rect_angle = 0

    rect = matplotlib.patches.Rectangle(rect_coord,
                                        rect_width,
                                        rect_height,
                                        rect_angle,
                                        edgecolor = 'black',
                                        facecolor = '#75D2FF',)
    axes.add_patch(rect)