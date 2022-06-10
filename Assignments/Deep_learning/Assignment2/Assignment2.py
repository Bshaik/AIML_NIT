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

""" Defining the paths """
MODELPATH = './model.h5'

"""Initializing the parameters"""
num_features = 64
num_labels = 7
batch_size = 64
epochs = 100
width, height = 48, 48

data = pd.read_csv('./fer2013.csv')
print(data.tail())

""" Pre-processing """
# Converting the relevant column element into a list for each row
pixels = data['pixels'].tolist()

faces = []
for pixel_sequence in pixels:
    # Splitting the string by space character as a list
    face = [int(pixel) for pixel in pixel_sequence.split(' ')]
    # Numpy
    face = np.asarray(face).reshape(width, height) 
    # Normalizing the image
    face = face / 255.0
    # Resizing the image
    face = cv2.resize(face.astype('uint8'), (width, height)) 
    faces.append(face.astype('float32'))

faces = np.asarray(faces)
# Expanding the dimension of channel for each image
faces = np.expand_dims(faces, -1) 
# Converting the labels to catergorical matrix
emotions = pd.get_dummies(data['emotion']).values

""" Split my model into training, validation and test sets"""
X_train, X_test, y_train, y_test = train_test_split(faces, emotions, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

""" Designing the CNN """
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

# Printing total trainable / non-trainable parameters
print(model.summary())

"""compile the model."""

model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

"""As long as the model continues to learn something, we continue to train the model to memorize (over-fit) the training data.
We stop the training of the model, if there is no change in the value of the loss function on the validation set for a certain epoch (patience).
"""
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

# To save the model during training
checkpointer = ModelCheckpoint(MODELPATH, monitor='val_loss', verbose=1, save_best_only=True)

""" Train the model"""
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

# Question: Do you get the same results if you run the Notebook multiple times without changing any parameters?
""" NO, the result varies"""

# Question: What is the effect of adding more neurons to each Conv2D layer?
""" The model gets more adaptive, so it can learn smaller details. But this means not necessary, that the classifier is then better on the 'next dataset'. 
The model gets more affected to over-fitting and so the generalization of your classification model can also decrease, 
e.g. this classifier will work worse on the next dataset."""

# Question: What happens if we manipulate the value of dropout?
""" With dropout (dropout rate less than some small value), the accuracy will gradually increase, and loss will gradually decrease first. 
When you increase dropout beyond a certain threshold, it results in the model not being able to fit properly. 
Intuitively, a higher dropout rate would result in a higher variance to some of the layers, which also degrades training."""

# Question: What is the effect of adding more activation layers to the network?
""" Adding layers increases the number of weights in the network, therefore the model complexity. 
Without a large training set, an increasingly large network is likely to overfit and in turn reduce accuracy on the test data."""

# Question: What is the accuracy score if we use more dense layers in the model?
"""More dense layers in the model increases accuracy and at the same time the complexity. 
An Increasingly large network is likely to overfit and in turn reduce accuracy on the test data."""

# Question: Does manipulating the learning rate affect the model? Justify your answer.
"""The learning rate is a configurable hyperparameter used in the training of neural networks 
that has a small positive value, ranges between 0.0 and 1.0. When the learning rate is too large, 
gradient descent can unintentionally increase rather than decrease the training error. When the learning rate is too small, 
training is not only slower, but may become permanently stuck with a high training error."""

# Question: Summary explaining how your program works
""" Here is the summary of the program
        1.	First and foremost: Importing the libraries
        2.	Define the model file path.
        3.	Read the data with the help of “pandas” from the csv file
        4.	Parameters Initialization: 
            where I feed the convolutional neural network with the images as batch, 
             which contains 64 images for each, in 100 epochs and eventually, 
             the network model will output the possibilities of 7 different emotions (num_classes) 
             can belong to the faces on the images sized with 48x48.
        5.	Preprocessing on the data:
            •	Converting the relevant column element into a list for each row
            •	Splitting the string by space character as a list
            •	Numpy
            •	Normalizing the image
            •	Resizing the image
            •	Expanding the dimension of channel for each image
            •	Converting the labels to categorical matrix
        6.	Split my model into training, validation and test sets
        7.	Designing the CNN
            •	In the first convolutional layer, L2 regularization (0.01) has been added.
            •	In all convolutional layers except the first one, batch normalization layer has been added.
            •	MAXP (2x2) and DROPOUT (0.25) layers have been added to each convolutional layer’s block.
            •	“RELU” has been picked as activation function for all convolutional layers.
        8.	Printing total trainable / non-trainable parameters
        9.	Now Compile the model. The categorical_crossentropy function has been picked out as a loss function because we have more than 2 labels and already prepared the labels in the categorical matrix structure.
        10.	Added some more features to our model
            •	If there is no improvement on the value of the loss function for the validation set after a certain epoch (patience) then to get rid of the “plateaus” it will reduce the learning rate parameter with a certain value.
            •	As long as the model continues to learn something, we continue to train the model to memorize (over-fit) the training data. It will stop the training of the model, if there is no change in the value of the loss function on the validation set for a certain epoch (patience).
            •	Finally, saving the model during training as long as it gets a better result than the previous epoch. Thus, we will have the best possible model at the end of the training.
        11.	Train the model
        12.	Measuring the performance of the model on the test set.
            •	With these parameters finally the performance of this model I am able to achieve the accuracy of around 64%"""
