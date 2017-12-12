import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.preprocessing import image
# from keras.applications.xception import Xception
# from keras.models import Model
# from keras.layers import Dense, Dropout, Flatten
# from keras.callbacks import EarlyStopping
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
size = (80, 80)

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

targets_series = pd.factorize(pd.Series(y))

x = np.array(x, np.float32)
y = np.array(targets_series[0], np.int_)

print(x.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1001)

print(y_test.shape)

feature_columns = [tf.feature_column.numeric_column("x", shape=x.shape)]

num_class = len(set(y))

model = tf.estimator.DNNClassifier(
  hidden_units  = [10, 20, 10],
  feature_columns=feature_columns,
  n_classes=num_class,
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    num_epochs=None,
    shuffle=True)

model.train(train_input_fn, steps=2000)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)

accuracy_score = model.evaluate(input_fn=test_input_fn)["accuracy"]

print(f'Acc: {accuracy_score}')