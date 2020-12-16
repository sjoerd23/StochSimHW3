import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import ast 

fname = "test1.csv"

df = pd.read_csv("results/" + fname)


def find_min_list_idx(l):
    list_len = [len(i) for i in l]
    return np.argmin(np.array(list_len))

x = ast.literal_eval(df["Data"][0])

for i in range(len(df)):
	run_data = ast.literal_eval(df["Data"][i])
	ind = find_min_list_idx(run_data)
	length = len(run_data[ind])
	for j in range(len(run_data)):
		full_data = run_data[j]
		run_data[j] = full_data[:length]
	# print(type(run_data))
	# print(run_data[0])
	mean = np.mean(run_data, axis=0)
	print("okay")
	confint = 1.96 * np.std(run_data, ddof=1, axis=0) / np.sqrt(len(run_data))
	x = np.linspace(0, len(run_data[0]), len(run_data[0]))
	plt.errorbar(x, y = mean, yerr = confint, label='M: {}, t0: {}'.format(df["Markov Length"]\
		[i], df["T0"][i]) )
	# for j in range(len(run_data)):
	# 	ind_run = run_data[j]
	# 	plt.plot(ind_run)
plt.legend()
plt.show()	