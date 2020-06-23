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
x_wav = np.load("./Data/Wave_angle.npy")
x_not_wav = np.load("./Data/Not_Wave_angle.npy")
y_wav = np.zeros(648)
y_not_wav = np.ones(650)
x_raw = np.append(x_wav, x_not_wav, axis=0)
y_raw = np.append(y_wav, y_not_wav, axis=0)
x_raw = x_raw.reshape(x_raw.shape[0],x_raw.shape[2],x_raw.shape[1])
#%%
x_new = []
for sample in x_raw:
    sample_new = []
    for i in range(0, 300, 6):
        sample_new.append(sample[i])    
    x_new.append(sample_new)
x_new = np.array(x_new)
#%%
# x_train = x_new.reshape(x_new.shape[0],x_new.shape[1],x_new.shape[2]*x_new.shape[3])
x_train = x_new

#%%

y_train = y_raw
print(x_train.shape)
print(y_train.shape)
#%%
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
y_train = keras.utils.to_categorical(y_train, num_classes=2, dtype='float32')
y_valid = keras.utils.to_categorical(y_valid, num_classes=2, dtype='float32')
np.save("x_test.npy",x_test)
np.save("y_test.npy",y_test)
print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)

# %%
epochs = 350
model = Sequential()

inputs = Input([50, 4])

# input_shortcut = inputs
conv1_1 = Conv1D(filters=128, kernel_size=10,padding = 'same', activation='relu')(inputs)
input_shortcut = conv1_1

conv1 = Conv1D(filters=128, kernel_size=10,padding = 'same', activation='relu')(inputs)
batch1 = BatchNormalization()(conv1)
drop1 = Dropout(0.3)(batch1)
conv2 = Conv1D(filters=128, kernel_size=10,padding = 'same', activation='relu')(drop1)
batch2 = BatchNormalization()(conv2)
drop2 = Dropout(0.3)(batch2)
conv3 = Conv1D(filters=128, kernel_size=10,padding = 'same')(drop2)
batch3 = BatchNormalization()(conv3)
drop3 = Dropout(0.5)(batch3)
res1 = Add()([input_shortcut,drop3])
res1_1 = res2_1 = Activation('relu')(res1)

# max1 = MaxPooling1D(pool_size=(3))(res1_1)

# conv4_1 = Conv1D(filters=128, kernel_size=30,padding = 'same', activation='relu')(max1)
# input_shortcut2 = conv4_1

# conv4 = Conv1D(filters=128, kernel_size=1,padding = 'valid', activation='relu')(max1)
# batch4 = BatchNormalization()(conv4)
# drop4 = Dropout(0.3)(batch4)
# conv5 = Conv1D(filters=128, kernel_size=30,padding = 'same', activation='relu')(drop4)
# batch5 = BatchNormalization()(conv5)
# drop5 = Dropout(0.3)(batch5)
# conv6 = Conv1D(filters=128, kernel_size=1,padding = 'valid')(drop5)
# batch6 = BatchNormalization()(conv6)
# drop6 = Dropout(0.5)(batch6)
# res2 = Add()([input_shortcut2,drop6])
# res2_1 = Activation('relu')(res2)

global1 = GlobalAveragePooling1D()(res1_1)
dense1 = Dense(128, activation='relu')(global1)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(2, activation='softmax')(dense3)
model = Model(inputs=inputs, outputs=dense4)
# model = Sequential()
# model.add(Conv1D(128, kernel_size = 10, activation='relu',input_shape = (x_train.shape[1:])))
# model.add(Conv1D(128, kernel_size = 10, activation='relu'))
# model.add(GlobalAveragePooling1D())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(2, activation='relu'))
model.compile(loss = "categorical_crossentropy", optimizer =Adam(learning_rate=1e-3), metrics= ["accuracy"])
model.summary()
#%%

history = model.fit(x_train,y_train, validation_data=(x_valid, y_valid), batch_size=32, epochs= epochs)
model.save("./Conv1d_Result/Conv1d_handwave_angle.h5")
# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred,axis=1)
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat = mat, show_normed = False, figsize = (9,9))
plt.savefig("./Conv1d_Result/Conf_Final.jpg")

# %%
