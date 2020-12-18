# plot.py
# used to plot the convergence plots created with tsp.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast
from collections import OrderedDict
import matplotlib.pyplot as plt

# define plotting colours
colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink",
"tab:gray", "tab:olive", "tab:cyan"]


# a function that return the index of the shortest nested list
def find_min_list_idx(l):
    list_len = [len(i) for i in l]
    return np.argmin(np.array(list_len))


# plotting the convergence for the different cooling schedules
fname = "total-test.csv"
df = pd.read_csv("results/" + fname)
x = ast.literal_eval(df["Data"][0])
labels = ["Logarithmic", "Linear", "Quadratic"]
plt.figure()

for i in range(len(df)):
    # load all runs for the specific parameters
    run_data = ast.literal_eval(df["Data"][i])

    # find the shortest run to truncate the rest to this length
    ind = find_min_list_idx(run_data)
    length = len(run_data[ind])
    for j in range(len(run_data)):
        full_data = run_data[j]
        run_data[j] = full_data[:length]

    # compute the mean and confidence interval
    mean = np.mean(run_data, axis=0)
    confint = 1.96 * np.std(run_data, ddof=1, axis=0) / np.sqrt(len(run_data))
    x = np.linspace(0, len(run_data[0]), len(run_data[0]))
    plt.errorbar(x, y=mean, yerr=confint, label='Cooling: {}'.format(labels[i]))

plt.xlabel("Iterations [-]")
plt.ylabel("Mean distance [-]")
plt.legend(fontsize=14)
axes = plt.gca()
axes.xaxis.label.set_size(14)
axes.yaxis.label.set_size(14)

# plot the convergence of the markov chain multiplier for some values of t0
fname = "logmark-large.csv"
df = pd.read_csv("results/" + fname)
x = ast.literal_eval(df["Data"][0])
t0s = list(df["T0"].unique())
mls = list(df["Markov Length"].unique())
plt.figure()

for i in range(len(df)):
    run_data = ast.literal_eval(df["Data"][i])
    min_dist = [np.min(run_data[j]) for j in range(len(run_data))]
    mean = np.mean(min_dist)
    confint = 1.96 * np.std(min_dist, ddof=1) / np.sqrt(len(min_dist))

    # extract the values for t0 and markov chain multiplier
    t0 = df["T0"][i]
    ml = df["Markov Length"][i]

    # select plotting colour
    c_index = t0s.index(t0)
    plt.errorbar(ml, y=mean, yerr=confint, marker='.', fmt='.', solid_capstyle="projecting",
        capsize=5, c=colors[c_index], label='t0 = {}'.format(t0))

plt.xlabel("Markov multiplier [-]")
plt.ylabel("Best distance [-]")

# remove duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# plot the convergence of t0 for some values of the markov chain multiplier
fname = "logt0.csv"
df = pd.read_csv("results/" + fname)
x = ast.literal_eval(df["Data"][0])
t0s = list(df["T0"].unique())
mls = list(df["Markov Length"].unique())
plt.figure()

for i in range(len(df)):
    run_data = ast.literal_eval(df["Data"][i])
    min_dist = [np.min(run_data[j]) for j in range(len(run_data))]

    mean = np.mean(min_dist)
    confint = 1.96 * np.std(min_dist, ddof=1) / np.sqrt(len(min_dist))
    ml = df["Markov Length"][i]
    t0 = df["T0"][i]

    c_index = mls.index(ml)
    if t0 != 0:
        plt.errorbar(t0, y=mean, yerr=confint, marker='.', fmt='.', solid_capstyle="projecting",
             capsize=5, c=colors[c_index], label='Markov multiplier = {:.1f}'.format((ml)))

plt.xlabel("T0 [-]")
plt.ylabel("Best distance [-]")
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()
