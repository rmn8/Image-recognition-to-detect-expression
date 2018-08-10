import keras
import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.misc import imread
from scipy.misc import imresize
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, InputLayer,Dropout
from keras.models import load_model,model_from_yaml
import math
train_set = pd.read_csv('/home/ram/Desktop/Image_rec/data/train.txt')
test_set = pd.read_csv('/home/ram/Desktop/Image_rec/data/test.txt')
train_count=train_set['Expression'].value_counts()
cl_weight={}
total=np.sum(train_count)

for i in range(0,len(train_count)):
	cl_weight[i]=math.log1p(15*total/float(train_count[i]))
cl_weight[7]=3.
print train_count
print cl_weight
temp=[]

for img_path in train_set.path:
	img = imread(img_path)
	img = imresize(img, (32, 32))
	img = img.astype('float32') 
	temp.append(img)


train_x = np.stack(temp)

temp=[]
no_of_output=train_set.Expression.value_counts().shape[0]
for img_path in test_set.path:
	img = imread(img_path)
	img = imresize(img, (32, 32))
	img = img.astype('float32') 
	temp.append(img)


test_x = np.stack(temp)
train_x = train_x / 255.
test_x = test_x / 255.


lb = LabelEncoder()
train_y = lb.fit_transform(train_set.Expression)
train_y = keras.utils.np_utils.to_categorical(train_y)
input_num_units = (32, 32, 3)
hidden_num_units = 512
output_num_units = no_of_output



epochs = 10
batch_size = 128

model = Sequential([
  InputLayer(input_shape=input_num_units),
  Flatten(),
  Dense(units=hidden_num_units, activation='relu'),
  Dropout(0.2),
  Dense(units=hidden_num_units, activation='relu'),
  Dropout(0.2),
  Dense(units=output_num_units, activation='softmax'),
])

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=batch_size,epochs=100,verbose=1, validation_split=0.2,class_weight=cl_weight)

pred = model.predict_classes(test_x)
pred = lb.inverse_transform(pred)

test_set['Class'] = pred
test_set.to_csv('../data/pred.csv', index=False)


"""
i = random.choice(train_set.index)
img_name = train_set.path[i]

img = imread(img_name)
pred = model.predict_classes(train_x)
print('Original:', train_set.Expression[i], 'Predicted:', lb.inverse_transform(pred[i]))

plt.imshow(imresize(img, (128, 128)))
plt.show()
"""
