import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# load labels
train = pd.read_csv('./data/boneage-training-dataset.csv')
test = pd.read_csv('./data/boneage-test-dataset.csv', sep=';')

print(test)

train_path = './data/boneage-training-dataset/boneage-training-dataset/post'
train_size = 1000
X_train = []
y_train = []

# load train set
for img in os.listdir(train_path):
    if len(X_train) == train_size:
        break
    image = Image.open(train_path+"/"+img)
    # To już jest w preprocessingu:
    #image = ImageOps.grayscale(image)
    image = np.array(image)
    image = image.reshape((512, 512, 1))
    X_train.append(image / 255)
    y_train.append(train.loc[lambda df: df['id'] == int(img.split('.')[0])].boneage.values[0])

X_train = np.array(X_train)
y_train = np.array(y_train)

# load test set

test_path = './data/boneage-test-dataset/boneage-test-dataset/post'
X_test = []
y_test = []

for img in os.listdir(test_path):
    image = Image.open(test_path+"/"+img)
    #image = ImageOps.grayscale(image)
    #image = image.resize((512,512))
    image = np.array(image)
    image = image.reshape((512, 512, 1))
    
    X_test.append(image / 255)
    y_test.append(test.loc[lambda df: df['id'] == int(img.split('.')[0])].boneage.values[0])

X_test = np.array(X_test)
y_test = np.array(y_test)

# create model

model = models.Sequential()
model.add(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(512, 512, 1)))
model.add(layers.MaxPooling2D((3,3)))
# model.add(layers.Conv2D(1, (5, 5), activation='relu'))
# model.add(layers.MaxPooling2D((5,5)))
# model.add(layers.Conv2D(1, (5, 5), activation='relu'))
# model.add(layers.MaxPooling2D((5,5)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.mean_absolute_error,
              metrics=['mean_absolute_error'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
print(pd.DataFrame({'prediction': model.predict(X_test).tolist(), 'boneage': y_test.tolist()}))