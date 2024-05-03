import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np



def draw_interval_and_f(f, x_lb : np.ndarray, x_ub : np.ndarray, y_lb : np.ndarray, y_ub : np.ndarray):
  plt.xlim(min(x_lb) - 0.5, max(x_ub) + 0.5)
  plt.ylim(min(y_lb) - 0.5, max(y_ub) + 0.5)
  colors = ["b", "g", "o", "r"]
  axes = plt.gca()
  axes.set_aspect("equal")
  x = np.arange(min(x_lb) - 0.1, max(x_ub) + 0.5, 0.01)
  for ff, c in zip(f, colors[:len(f)]):
    fxx = np.array([ff(i) for i in x])
    axes.plot(x, fxx, )
  for i in range(len(x_lb)):
    drawRect(axes, x_lb[i], x_ub[i], y_lb[i], y_ub[i])
  
  #plt.show()

def draw_interval_and_many_f(f_list, x_lb : np.ndarray, x_ub : np.ndarray, y_lb : np.ndarray, y_ub : np.ndarray):
  plt.xlim(min(x_lb) - 0.5, max(x_ub) + 0.5)
  plt.ylim(min(y_lb) - 0.5, max(y_ub) + 0.5)
  axes = plt.gca()
  axes.set_aspect("equal")
  for i in range(len(x_lb)):
    drawRect(axes, x_lb[i], x_ub[i], y_lb[i], y_ub[i])
  x = np.arange(min(x_lb) - 0.5, max(x_ub) + 0.5, 0.1)
  for f in f_list:
    fxx = np.array([f(i) for i in x])
    axes.plot(x, fxx)
  #plt.show()

def draw_interval( x_lb : np.ndarray, x_ub : np.ndarray, y_lb : np.ndarray, y_ub : np.ndarray):
  plt.xlim(min(x_lb) - 0.5, max(x_ub) + 0.5)
  plt.ylim(min(y_lb) - 0.5, max(y_ub) + 0.5)
  plt.rcParams ['figure.figsize'] = [10, 4]
  axes = plt.gca()
  axes.set_aspect("equal")
  for i in range(len(x_lb)):
    drawRect(axes, x_lb[i], x_ub[i], y_lb[i], y_ub[i])
  x = np.arange(min(x_lb) - 0.5, max(x_ub) + 0.5, 0.1)
  #plt.show()

def draw_many(a, b, x_lb : np.ndarray, x_ub : np.ndarray, y_lb : np.ndarray, y_ub : np.ndarray ):
  plt.xlim(min(x_lb) - 0.5, max(x_ub) + 0.5)
  plt.ylim(min(y_lb) - 0.5, max(y_ub) + 0.5)
  x = np.arange(min(x_lb) - 0.5, max(x_ub) + 0.5, 0.05)
  axes = plt.gca()
  axes.set_aspect("equal")
  for a_, b_ in zip(a, b):
    f = lambda x: a_ @ np.exp(-x * b_)
    fxx = np.array([f(i) for i in x])
    axes.plot(x, fxx)

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
                                        facecolor = '#75D2AA',)
    axes.add_patch(rect)

def show():
   plt.show()