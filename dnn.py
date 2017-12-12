import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
import os

def normalize(arr):
  arr = arr.astype('float')
  for i in range(3):
    minval = arr[...,i].min()
    maxval = arr[...,i].max()
    if minval != maxval:
      arr[...,i] -= minval
      arr[...,i] *= (255.0/(maxval-minval))
  return arr

train_path = './data/train/'
test_path = './data/ten/'
size = (60, 60)

x = []
y = []

df_test = pd.read_csv('./data/sample_submission.csv')

labels = []
for f in os.listdir('./data/train'):
  labels.append(f)

for label in os.listdir(train_path):
  if os.path.isdir(train_path + label):
    for f in os.listdir(train_path + label):
      img = image.load_img(train_path + label + '/' + f)
      img = img.resize(size)
      img = np.array(image.img_to_array(img))
      img = normalize(img)
      x.append(img)
      y.append(label)

print(labels)

targets_series = pd.Series(y)

x = np.array(x, np.float32)
y = np.array(pd.factorize(targets_series)[0], np.int_)

print(x.shape)
print(y.shape)

