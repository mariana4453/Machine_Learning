import math, copy
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from matplotlib import style
# print(plt.style.available)

from plot_functions import plotting

path = "./01_data/Activities.csv"
df_activities = pd.read_csv(path)

columns_rename = {'Тип занятия': 'activity',
                  'Дата': 'date',
                  'Название': 'name',
                  'Расстояние': 'Distance',
                  'Калории': 'Calories',
                  'Время': 'Time_of_activity',
                  'Средняя ЧП': 'Average_pulse',
                  'Максимальная ЧП': 'Maximum_pulse',
                  'Средняя частота шага': 'Average_pace',
                  'Максимальная частота шага': 'Maximum_pace',
                  'Средний темп': 'Average_speed',
                  'Лучший темп': 'Best_speed',
                  'Общий подъем': 'uphill',
                  'Общий спуск': 'downhill',
                  'Training Stress Score®': 'training_stress_score',
                  'Время лучшего круга': 'Best_lap_time',
                  'Количество кругов': 'laps'
                  }
df_activities = df_activities.rename(columns=columns_rename)
df_activities = df_activities[['activity', 'date', 'name', 'Distance', 'Calories', 'Time_of_activity',
                               'Average_pulse', 'Maximum_pulse', 'Average_pace', 'Maximum_pace', 'Average_speed',
                               'Best_speed', 'uphill', 'downhill', 'training_stress_score', 'Best_lap_time', 'laps']]



# GRADIENT DESCENT
df_for_prediction = df_activities[['Distance', 'laps']]
x_train = df_for_prediction['laps']
y_train = df_for_prediction['Distance']

# plt.scatter(x_train, y_train)
# plt.show()


# 1 - cost function
w = 10
b = 15

def cost_calculation(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb_i = w * x[i] + b
        f_wb_i = (f_wb_i - y[i]) ** 2
        cost += f_wb_i

    total_cost = 1 / (2 * m) * cost
    return total_cost


def gradient_calcucation(x, y, w, b):
    m = x_train.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb_i = w * x[i] + b
        f_wb_i = f_wb_i - y[i]

        dj_dw += f_wb_i * x[i]
        dj_db += f_wb_i

    dj_dw = (1/m) * dj_dw
    dj_db = (1/m) * dj_db
    return dj_dw, dj_db


# PLOTTING 1
w_array = np.arange(-5, 21, 1)
# plotting(w_array, x_train, y_train, b, cost_calculation, gradient_calcucation)


def gradient_descent(x, y, w_init, b_init, alpha, num_iters, cost_function, gradient_function):

    # w = copy.deepcopy(w_init)
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []

    b = b_init
    w = w_init

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update Parameters
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # if i % 20 == 0:
            # print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
            #       f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
            #       f"w: {w: 0.3e}, b:{b: 0.5e}")

            # print("Iteration: {}; Cost: {}; dj_dw: {}; dj_db: {}; w: {}; b: {}".format(i, J_history[-1], dj_dw, dj_db, w, b))
    return w, b, J_history, p_history

w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init=5, b_init=100, alpha=0.01, num_iters=1000, cost_function=cost_calculation, gradient_function=gradient_calcucation)


# PLOTTING 2
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
# ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step')

plt.show()