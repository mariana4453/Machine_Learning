import math, copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


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


# trying Linear regression - one variable
df_sample = df_activities[['Distance', 'laps']]

x_train = df_sample['Distance']
y_train = df_sample['laps']

# Model function: 𝑓 𝑤,𝑏( 𝑥(𝑖) ) = 𝑤𝑥(𝑖)+𝑏
# trying:
# for  𝑥(0),f_wb = w * x[0] + b
w = 1
b = 1


def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb


# plotting the result
# tmp_f_wb = compute_model_output(x_train, w, b)
#
# # prediction
# plt.plot(x_train, tmp_f_wb, c='blue', label='Prediction')
#
# # Plot the data points
# plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
#
# # Set the title
# plt.title("Housing Prices")
#
# # Set the y-axis label
# plt.ylabel('Price (in 1000s of dollars)')
#
# # Set the x-axis label
# plt.xlabel('Size (1000 sqft)')
# plt.legend()
# plt.show()


# cost function - the lesser the cost the better it is
# 𝐽 (𝑤,𝑏) = 1/2𝑚 ∑ (𝑓𝑤,𝑏 (𝑥(𝑖)) − 𝑦(𝑖) ) ^ 2

# w = np.arange(-10, 10, 1)

def compute_cost(x, y, w, b):
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


# total_cost_list = []
# for i in range(w.shape[0]):
#     cost = compute_cost(x_train, y_train, i, b)
#     total_cost_list.append(cost)
#
# x = w
# y = total_cost_list
#
# plt.plot(x, y)
# plt.show()