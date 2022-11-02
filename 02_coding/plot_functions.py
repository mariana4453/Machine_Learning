import math, copy
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from matplotlib import style
plt.style.use('dark_background')
# print(plt.style.available)


def add_line(dj_dx, x1, y1, d, ax):
    x = np.linspace(x1-d, x1+d,50)
    y = dj_dx*(x - x1) + y1
    ax.scatter(x1, y1, color='red', s=50)
    ax.plot(x, y, '--', c='darkgreen', zorder=10, linewidth = 1)
    xoff = 30 if x1 == 200 else 10
    ax.annotate(r"$\frac{\partial J}{\partial w}$ =%d" % dj_dx, fontsize=10,
                xy=(x1, y1), xycoords='data',
            xytext=(xoff, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment='left', verticalalignment='top')



def plotting(w_array, x, y, b, cost_calculation, gradient_calcucation):
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    cost = np.zeros_like(w_array)

    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = cost_calculation(x, y, tmp_w, b)

    ax.plot(w_array, cost, linewidth=1)
    ax.set_title("Cost vs w, with gradient; b set to {}".format(b))
    ax.set_ylabel('Cost')
    ax.set_xlabel('w')

    # plot lines for fixed b
    for tmp_w in [-3, 0, 5, 10, 15]:
        dj_dw, dj_db = gradient_calcucation(x, y, tmp_w, b)
        j = cost_calculation(x, y, tmp_w, b)
        add_line(dj_dw, tmp_w, j, 10, ax)

    plt.show()
