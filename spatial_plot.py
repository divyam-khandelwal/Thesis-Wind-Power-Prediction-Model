import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import math
from itertools import islice
import sys
import operator
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib
import os

# Column formats
headers = ["Year", "Month", "Day", "Hour", "Minute",
           "Power"]

data_i = pd.read_csv("SpatialCorrelationData/FirstFarm/9074-2012.csv", sep=',', skiprows=4, names=headers)
# print(data.head)

headers_err = ["Year", "Month", "Day", "Hour", "Minute",
           "Power", "Air Temperature"]
data_ii = pd.read_csv("SpatialCorrelationData/SecondFarm/9073-2012.csv", sep=',', skiprows=4, names=headers_err)


req_resolution = 0.1 # in (MW)

max_wind_power = data_i['Power'].max()
min_wind_power = data_i['Power'].min()

# Discretise wind power
n_bins = int(math.ceil((max_wind_power - min_wind_power) / req_resolution))

order = 2
# Construct transition matrix
kernel_mtx = [[0.0 for i in range((n_bins + 1)**order)] for j in range((n_bins + 1)**order)]

bin_size = (max_wind_power - min_wind_power) / n_bins

prev_state_idx_i, prev_state_idx_ii = [None for _ in range(order)], [None for _ in range(order)]

for i in range(0, order):
    power_state_i = data_i['Power'][i]
    power_state_ii = data_ii['Power'][i]

    prev_state_idx_i[i] = int(power_state_i / bin_size)
    prev_state_idx_ii[i] = int(power_state_ii / bin_size)

curr_state_idx_i, curr_state_idx_ii = [None for _ in range(order)], [None for _ in range(order)]

for i in range(order, len(data_i) - 1):

    curr_state_idx_i[0] = int(data_i['Power'][i] / bin_size)
    curr_state_idx_ii[0] = int(data_ii['Power'][i] / bin_size)

    # Calculate previous index
    temp_prev = prev_state_idx_i[0] * n_bins + prev_state_idx_ii[0]

    # Calculate current index
    temp_curr = curr_state_idx_i[0] * n_bins + curr_state_idx_ii[0]

    kernel_mtx[temp_prev][temp_curr] += 1

    # Update previous index
    prev_state_idx_i[0] = curr_state_idx_i[0]
    prev_state_idx_ii[1] = curr_state_idx_i[0]

# Normalise
for j in range(len(kernel_mtx)):
            norm_sum = 0
            for elem in kernel_mtx[j]:
                norm_sum += elem
            if norm_sum != 0:
                kernel_mtx[j] = [x / norm_sum for x in kernel_mtx[j]]

ax = seaborn.heatmap(kernel_mtx)