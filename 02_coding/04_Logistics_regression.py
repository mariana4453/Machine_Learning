import copy
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
# from additional.plt_one_addpt_onclick import plt_one_addpt_onclick
from additional.lab_utils_common import draw_vthresh
from additional.lab_utils_common import plot_data

plt.style.use('seaborn-dark')
# print(plt.style.available)

# sigmoid function
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g
#
# z_tmp = np.arange(-10, 11)
# y = sigmoid(z_tmp)

#
# np.set_printoptions(precision=3)
# print("Input (z), Output (sigmoid(z))")
# print(np.c_[z_tmp, y])


# Plot z vs sigmoid(z)
# fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# ax.plot(z_tmp, y, c="b")
#
# ax.set_title("Sigmoid function")
# ax.set_ylabel('sigmoid(z)')
# ax.set_xlabel('z')
# draw_vthresh(ax, 0)
#
# plt.show()
# plt.close('all')


# applying example - 1
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.array([2.,3.])
b_tmp = 1.

#  plotting
# fig, ax = plt.subplots(1, 1, figsize=(4,4))
# plot_data(X_train, y_train, ax)
#
# ax.axis([0, 4, 0, 3.35])
# ax.set_ylabel('$x_1$', fontsize=12)
# ax.set_xlabel('$x_0$', fontsize=12)
# plt.show()


# function gradient
def lr_compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]

        for j in range(n):
            dj_dw_j = err_i * X[i, j]
            dj_dw[j] += dj_dw_j

        dj_db += err_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw


# cost
def compute_cost(X, y, w, b):
    m, n = X.shape
    J_cost = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w_tmp) + b_tmp)
        cost = -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
        J_cost += cost

    J_cost = J_cost / m
    return J_cost




alph = 0.1
iters = 10

# compute gradient descent
def gradient_descent(X, y, w_init, b_init, alpha, num_iters):
    J_history = []

    w = w_init
    b = b_init

    for i in range(num_iters):
        dj_db, dj_dw = lr_compute_gradient(X, y, w, b)

        w = w_init
        b = b_init

        J_history.append(compute_cost(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history
