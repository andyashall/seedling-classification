import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

base_model = Xception(
  weights = 'imagenet',
  include_top=False,
  input_shape=(80, 80, 3)
)

num_class = len(set(y))

print(num_class)

# Add a new top layer
out = base_model.output
out = Flatten()(out)
predictions = Dense(num_class, activation='softmax')(out)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(
  loss='categorical_crossentropy', 
  optimizer='adam', 
  metrics=['accuracy']
)

callbacks_list = [EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

model.summary()

model.fit(X_train, y_train, epochs=1, validation_data=(X_test, y_test), verbose=1)

preds = model.predict(X_test, verbose=1)

acc = accuracy_score(y_test, preds)

print(acc)

# sub = pd.DataFrame(preds)

# sub.columns = one_hot.columns.values

# sub['id'] = test_ids

print(sub.head(5))