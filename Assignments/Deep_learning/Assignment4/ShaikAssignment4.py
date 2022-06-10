# Importing required libraries

import os
from random import shuffle
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.utils.np_utils import to_categorical

# os.chdir('../input')
os.chdir('D:/ML_AI/NITW_PG_Soltions-master/Assignments/Deep_learning/Assignment4/input')
print(os.listdir())

# Checking the directory
os.chdir('UTKFace')

# verifying one of the image as a sample
im = Image.open('1_0_0_20161219140623097.jpg.chip.jpg').resize((128, 128))
im

onlyfiles = os.listdir()
print(len(onlyfiles))

shuffle(onlyfiles)
gender = [i.split('_')[1] for i in onlyfiles]

# Separating the labels from the images so that they are stored in the classes and spliting the data into Gender Classes as
# - 0 Male
# 1 Female
classes = []
for i in gender:
    i = int(i)
    classes.append(i)

# CONVERT IMAGES TO VECTORS
X_data = []
for file in onlyfiles:
    face = imageio.imread(file)
    face = cv2.resize(face, (128, 128))
    X_data.append(face)

X = np.squeeze(X_data)
print(X.shape)

# normalize data
X = X.astype('float32')
X /= 255

print(classes[:10])
categorical_labels = to_categorical(classes, num_classes=2)

print(categorical_labels[:10])

# Splitting the data
(x_train, y_train), (x_test, y_test) = (X[:15008], categorical_labels[:15008]), (X[15008:], categorical_labels[15008:])
(x_valid, y_valid) = (x_test[:7000], y_test[:7000])
(x_test, y_test) = (x_test[7000:], y_test[7000:])

len(x_train) + len(x_test) + len(x_valid) == len(X)

# Creating the model
model = tf.keras.Sequential()

model.add(
    tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(128, 128, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))

# To look at the model summary
print(model.summary())

# Compiling the Model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Model fit
model.fit(x_train,
          y_train,
          batch_size=64,
          epochs=5,
          validation_data=(x_valid, y_valid), )

# Saving the Model
model.save('./model2.h5')

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])

labels = ["Male",  # index 0
          "Female",  # index 1
          ]
print('Male ->', '0', '\nFemale ->', '1')

y_hat = model.predict(x_test)

# Plotting a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(labels[predict_index],
                                  labels[true_index]),
                 color=("green" if predict_index == true_index else "red"))
plt.show()
