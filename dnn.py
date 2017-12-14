import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from PIL import ImageFilter, ImageOps
from keras.applications.xception import Xception
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import os

size = (140, 140)

def pre_proc(img):
  img = img.resize(size)
  # img = img.filter(ImageFilter.FIND_EDGES)
  img = np.array(image.img_to_array(img))
  img *= 255.0/img.max()
  img[img[:, :, 1] < 100] = 0
  img[img[:, :, 2] > 50] = 0
  return img

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

# Try: mirror and flip training images
for label in os.listdir(train_path):
  if os.path.isdir(train_path + label):
    for f in os.listdir(train_path + label):
      img = image.load_img(train_path + label + '/' + f)
      imgm = ImageOps.mirror(img)
      imgm = pre_proc(imgm)
      img = pre_proc(img)
      x.append(img)
      y.append(label)
      x.append(imgm)
      y.append(label)

for f in os.listdir(test_path):
  img = image.load_img(test_path + '/' + f)
  img = pre_proc(img)
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

model.summary()

batch_size = 64
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
earlystop = EarlyStopping(monitor='val_acc', patience=10)
modelsave = ModelCheckpoint(filepath='model.h5', save_best_only=True, verbose=1)

model.fit(X_train, y_train, batch_size=batch_size, epochs=10, validation_data=(X_test, y_test), verbose=1, callbacks=[annealer, earlystop, modelsave])

# Predict with test data
preds = model.predict(test, verbose=1)

# Create and save submission file
sub = pd.DataFrame(preds)
sub.columns = one_hot.columns.values
sub['species'] = sub.idxmax(axis=1)
sub['file'] = test_ids
print(sub.head(5))

sub.to_csv('sub.csv', columns=['file', 'species'], index=False, float_format='%.3f')