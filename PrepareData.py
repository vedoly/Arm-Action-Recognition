#%%
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import *
from keras.optimizers import *
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
#%%
x_wav = np.load("./Data/onlyarm_handwave.npy")
x_not_wav = np.load("./Data/onlyarm_not_handwave.npy")
x_raw = np.append(x_wav, x_not_wav, axis=0)
# %%
x_new = []
for sample in x_raw:
    sample_new = []
    for i in range(0, 300, 6):
        sample_new.append(sample[i])    
    x_new.append(sample_new)
x_new = np.array(x_new)
# %%
j = []
for i in range(0, 300, 6):
    print(i)
    j.append(i)
    # if j == 294:
    #     break

# %%
data = np.load("data.npy")
# data2 = np.load("data 100.npy")
# %%
## 5-10