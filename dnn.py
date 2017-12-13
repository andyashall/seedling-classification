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

train_path = './data/train/'
test_path = './data/test/'
size = (140, 140)

x = []
y = []
test = []
test_ids = []

df_test = pd.read_csv('./data/sample_submission.csv')

labels = []
for f in os.listdir('./data/train'):
  labels.append(f)

for label in os.listdir(train_path):
  if os.path.isdir(train_path + label):
    for f in os.listdir(train_path + label):
      img = image.load_img(train_path + label + '/' + f)
      # image.ImageOps.mirror(img)
      # red, green, blue = img.split()
      img = img.resize(size)
      img = np.array(image.img_to_array(img))
      img *= 255.0/img.max()
      # img[:,:,0] *= 0
      # img[:,:,2] *= 0
      img[0,:,0] = 0
      x.append(img)
      y.append(label)

for f in os.listdir(test_path):
  img = image.load_img(test_path + '/' + f)
  img = img.resize(size)
  img = np.array(image.img_to_array(img))
  img *= 255.0/img.max()
  # img[:,:,0] *= 0
  # img[:,:,2] *= 0
  img[0,:,0] = 0
  test.append(img)
  test_ids.append(f)

test = np.array(test, np.float32)

print(labels)

targets_series = pd.Series(y)

one_hot = pd.get_dummies(targets_series, sparse = True)

x = np.array(x, np.float32)
y = np.array(one_hot, np.int_)

print(x.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=1001)

print(y_test.shape)

base_model = Xception(
  weights = 'imagenet',
  include_top=False,
  input_shape=(140, 140, 3)
)

num_class = y.shape[1]

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

preds = model.predict(test, verbose=1)

sub = pd.DataFrame(preds)

sub.columns = one_hot.columns.values

sub['species'] = sub.idxmax(axis=1)

sub['file'] = test_ids

print(sub.head(5))

sub.to_csv('sub.csv', columns=['file', 'species'], index=False, float_format='%.3f')