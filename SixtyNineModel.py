#importing libaries
import re
import pickle
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import os
import sys
import time
import random
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Dense, Dropout, Conv2D, Input, Lambda, Flatten, TimeDistributed, merge
from keras.layers import Add, Reshape, MaxPooling2D, Concatenate, Embedding, RepeatVector
from keras.models import Model
from keras.layers.core import Activation
from keras.utils import np_utils, to_categorical
import matplotlib.pyplot as plt

random.seed(31)

#define constants
max_dist = 1
len_feature = 69
n_phases = 3

classes = ['Climbing', 'Crusing', 'Descending']

train_df = pd.read_csv('SixtyNineTrain.csv')
train_df = train_df.sample(frac=1)

test_df = pd.read_csv('SixtyNineTest.csv')
test_df = test_df.sample(frac=1)


def observation(index, dataframe, class_list=classes, len_feat=len_feature):

	out = []
	for i in range(len(classes)):
		out.append(np.array(
			dataframe.iloc[index, (i*len_feat):((i+1) * len_feat)].values.flatten().tolist()))
	return out


def Adjacency(class_list):

	out = []
	for phase in class_list:
		tmp1 = []
		for cls in class_list:
			tmp = [0] * len(class_list)
			if abs(class_list.index(phase) - class_list.index(cls)) <= max_dist and abs(class_list.index(phase) - class_list.index(cls)) >= 0:
				tmp[class_list.index(cls)] = 1
			tmp1.append(tmp)
		out.append(tmp1)
	return out


def MLP():

	In_0 = Input(shape=[len_feature])
	h = Dense(128, activation='relu', kernel_initializer='random_normal')(In_0)
	#h = Dropout(0.4)(h)
	h = Dense(128, activation='relu', kernel_initializer='random_normal')(h)
	#h = Dropout(0.4)(h)
	h = Reshape((1, 128))(h)
	model = Model(input=In_0, output=h)
	return model


def MultiHeadsAttModel(l=2, d=128, dv=16, dout=128, nv=8):

	v1 = Input(shape=(l, d))
	q1 = Input(shape=(l, d))
	k1 = Input(shape=(l, d))
	ve = Input(shape=(1, l))

	v2 = Dense(dv*nv, activation="relu", kernel_initializer='random_normal')(v1)
	#v2 = Dropout(0.4)(v2)
	q2 = Dense(dv*nv, activation="relu", kernel_initializer='random_normal')(q1)
	#q2 = Dropout(0.4)(q2)
	k2 = Dense(dv*nv, activation="relu", kernel_initializer='random_normal')(k1)
	#k2 = Dropout(0.4)(k2)

	v = Reshape((l, nv, dv))(v2)
	q = Reshape((l, nv, dv))(q2)
	k = Reshape((l, nv, dv))(k2)
	v = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(v)
	k = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(k)
	q = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(q)

	att = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[
				 3, 3]) / np.sqrt(dv))([q, k])  # l, nv, nv
	att = Lambda(lambda x: K.softmax(x))(att)
	out = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[3, 2]))([att, v])
	out = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)))(out)

	out = Reshape((l, dv*nv))(out)

	T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([ve, out])

	out = Dense(dout, activation="relu", kernel_initializer='random_normal')(T)
	model = Model(inputs=[q1, k1, v1, ve], outputs=out)
	return model


def Regressor():

	I1 = Input(shape=(1, 128))
	I2 = Input(shape=(1, 128))
	I3 = Input(shape=(1, 128))

	h1 = Flatten()(I1)
	h2 = Flatten()(I2)
	h3 = Flatten()(I3)

	h = Concatenate()([h1, h2, h3])

	V = Dense(128, activation="relu", kernel_initializer='random_normal')(h)
	V = Dropout(0.2)(V)
	V = Dense(48, activation="relu", kernel_initializer='random_normal')(V)
	V = Dropout(0.2)(V)
	V = Dense(1, kernel_initializer='random_normal')(V)

	model = Model(input=[I1, I2, I3], output=V)
	return model

def my_tf_round(x, decimals):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def out_round(x):
	x_rounded_NOT_differentiable = my_tf_round(x, 1)
	x_rounded_differentiable = x - \
		tf.stop_gradient(x - x_rounded_NOT_differentiable)
	return x_rounded_differentiable

encoder = MLP()
m1 = MultiHeadsAttModel(l=n_phases)
m2 = MultiHeadsAttModel(l=n_phases)
reg = Regressor()
vec = np.zeros((1, n_phases))
vec[0][0] = 1

In = []
for j in range(n_phases):
	In.append(Input(shape=[len_feature]))
	In.append(Input(shape=(n_phases, n_phases)))
In.append(Input(shape=(1, n_phases)))

feature = []

for j in range(n_phases):
	feature.append(encoder(In[j*2]))

feature_ = Concatenate(axis=1)(feature)

relation1 = []
for j in range(n_phases):
	T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In[j*2+1], feature_])
	relation1.append(m1([T, T, T, In[n_phases*2]]))

relation1_ = Concatenate(axis=1)(relation1)

relation2 = []
for j in range(n_phases):
	T = Lambda(lambda x: K.batch_dot(x[0], x[1]))([In[j*2+1], relation1_])
	relation2.append(m2([T, T, T, In[n_phases*2]]))

V = []
for j in range(n_phases):
	V.append(reg([feature[j], relation1[j], relation2[j]]))

Out = Lambda(lambda x: tf.reduce_mean(x, axis=0))(V)
Rounded_Out = Lambda(out_round)(Out)

#mean of agents
model = Model(input=In, output=Rounded_Out)
model.compile(optimizer=Adam(lr=0.001), loss='mae')

Wsave = model.get_weights()


def create_train_val(train_per, df):

	val_len = int(len(df) * (1 - train_per))

	x_train = []
	x_val = []

	for i_ in range(n_phases*2 + 1):
		x_train.append([])
		x_val.append([])

	for i in range(len(df)):
		obs = observation(i, df)
		if i < len(df) - val_len:
			for j in range(n_phases):
				x_train[j*2].append(obs[j])
				x_train[j*2+1].append(adj[j])
			x_train[n_phases*2].append(vec)
		else:
			for j in range(n_phases):
				x_val[j*2].append(obs[j])
				x_val[j*2+1].append(adj[j])
			x_val[n_phases*2].append(vec)

	for i_ in range(n_phases*2+1):
		x_train[i_] = np.asarray(x_train[i_])
		x_val[i_] = np.asarray(x_val[i_])

	y_train = np.random.random((len(df) - val_len, 1))
	y_val = np.random.random((val_len, 1))

	for i in range(len(df)):
		if i < len(df) - val_len:
			y_train[i] = df.iloc[i, 207]
		else:
			y_val[val_len - len(df) + i] = df.iloc[i, 207]

	return x_train, y_train, x_val, y_val

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

adj = Adjacency(classes)

x_train, y_train, x_val, y_val = create_train_val(1, train_df)
x_test, y_test, x_dummy, y_dummy = create_train_val(1, test_df)

history = model.fit(x_train, y_train, epochs=50, batch_size=128,
                    validation_data=(x_test, y_test))  # , callbacks=[PredictionCallback()])

plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Best Model Results')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.savefig('SixtyNinelModelCI.png')

#model.save('SixtyNineModel.h5')
