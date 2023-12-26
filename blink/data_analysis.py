import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.stats import norm, lognorm
import warnings
import math

sns.set(context='notebook', font='simhei', style='whitegrid')
warnings.filterwarnings('ignore')


data = np.load('vec7/interval_svm.npy')
print(type(data))
s = []
logdata = []
for n in data:
    num = 1800 / n
    if num <= 100:
        s.append(num)
        logdata.append(np.log(num))
s = np.array(s)
logdata = np.array(logdata)

mu = np.mean(logdata)
sigma = np.std(logdata)
print(mu, sigma)
num_bins = 25
n, bins, patches = plt.hist(s, num_bins, density=True, color='grey')

plt.xlabel('Blinks per minute')
plt.ylabel('Relative Frequency')

plt.subplots_adjust(left=0.15)

x = np.linspace(0, 100, 20000)
plt.plot(x, lognorm.pdf(x=x, s=sigma, scale=math.exp(mu)), color="red")
plt.show()

# 采样
sample = np.random.lognormal(mu, sigma, 10000)
