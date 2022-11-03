import math, copy
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from matplotlib import style
# print(plt.style.available)
plt.style.use('grayscale')

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
df_model = df_activities[['Distance', 'laps', 'Average_pulse', 'Calories']]

x_train = df_model[['Distance', 'laps', 'Average_pulse']]
x_train = x_train.to_numpy()

y_train = df_model[['Calories']]
y_train = y_train['Calories'].str.replace(',', '').astype(int)

b = 10
w = np.array([1, 0.3, 2.5])


# calculation for single value (first row)
x_train_single = x_train[0]

def single_value_prediction(x, w, b):
    p = np.dot(x, w) + b
    return p

# print(single_value_prediction(x_train_single, w, b))

def cost_for_mult_var(x, y, w, b):
    m = x_train.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(x_train[i], w) + b
        cost += (f_wb - y_train[i]) ** 2
    final_cost = 1 / (2 * m) * cost
    return final_cost
