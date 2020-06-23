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

model = tf.keras.models.load_model("Conv1d.h5")
ScorePredict = []
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")
y_test = list(y_test)
y_test = [list(y_test[i]).index(1) for i in range(len(y_test))]
for i in range(x_test.shape[0]):
    kuy = x_test[i].reshape(1,300,75)
    ScoreDecoy = model.predict(kuy)
    result = np.where(ScoreDecoy == np.amax(ScoreDecoy))
    ScorePredict.append(result)
#%%
ScorePredict = [int(ScorePredict[i][1]) for i in range(len(ScorePredict))]
#%%
# Dict_compare =  dict(zip(ScorePredict, y_test))
# %%
count = 0
for i in range(len(ScorePredict)):
    if ScorePredict[i] == y_test[i] :
        count +=1

print("Predict action accurately :", count)
print("From %d sequence" %(len(y_test)))
# %%
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
y_pred = model.predict_classes(x_test)

mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat = mat, show_normed = False, figsize = (7,7))
# plot_confusion_matrix(model, x_test, y_test)
# %%
# from sklearn import metrics
# metrics.confusion_matrix