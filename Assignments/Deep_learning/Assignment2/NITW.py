import os
import sys

import cv2
import keras
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split

BASEPATH = './'
sys.path.insert(0, BASEPATH)
os.chdir(BASEPATH)
MODELPATH = './NITWmodel.h5'

num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

data = pd.read_csv('./fer2013.csv')
print(data.tail())

pixels = data['pixels'].tolist()  # 1

faces = []
for pixel_sequence in pixels:
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]  # 2
    face = np.asarray(face).reshape(width, height)  # 3

    # There is an issue for normalizing images. Just comment out 4 and 5 lines until when I found the solution.
    face = face / 255.0  # 4
    face = cv2.resize(face.astype('uint8'), (width, height))  # 5
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
faces = np.expand_dims(faces, -1)  # 6

emotions = pd.get_dummies(data['emotion']).values  # 7

X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

model = keras.Sequential()

model.add(Conv2D(num_features, kernel_size=(5, 5), activation='relu', input_shape=(width, height, 1),
                 data_format='channels_last', kernel_regularizer=l2(0.01)))
model.add(Conv2D(num_features, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(2 * num_features, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2 * num_features, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(2 * 2 * num_features, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2 * 2 * num_features, kernel_size=(5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2 * 2 * num_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2 * num_features, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(num_labels, activation='softmax'))

print(model.summary())

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

tensorboard = TensorBoard(log_dir='./logs')

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

checkpointer = ModelCheckpoint(MODELPATH, monitor='val_loss', verbose=1, save_best_only=True)

model.fit(np.array(X_train), np.array(y_train),
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(np.array(X_test), np.array(y_test)),
          shuffle=True,
          callbacks=[lr_reducer, early_stopper, checkpointer])

scores = model.evaluate(np.array(X_test), np.array(y_test), batch_size=batch_size)
print("Loss: " + str(scores[0]))
print("Accuracy: " + str(scores[1]))
