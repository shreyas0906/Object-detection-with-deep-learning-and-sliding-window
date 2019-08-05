# importing the required libraries
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import tensorflow as tf

img_rows, img_cols = 45,45 # Specifying the dimensions of the image
test_label = np.load('testing_img_label.npy')# loading the training and testing data
test_images = np.load('testing_img.npy')
train_label = np.load('training_img_labels.npy')
train_imgs = np.load('training_img.npy')
train_label = np_utils.to_categorical(train_label, 2)# converting the image labels to categories
test_label = np_utils.to_categorical(test_label, 2)


def preprocess(imgs): # reshaping the image
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)/255
    return imgs_p


train_imgs = preprocess(train_imgs)
test_images = preprocess(test_images)
# The neural network
epochs = 5
model = Sequential()
model.add(Convolution2D(225,2,2,input_shape=(1,45,45),border_mode='same',activation='relu',subsample=(1,1),W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Convolution2D(128,2,2,activation='relu',border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.3)) # adding the dropout model to improve trainign

model.add(Convolution2D(64,2,2,activation='relu',border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(64,activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.5))

model.add(Dense(32,activation='relu',W_constraint=maxnorm(3)))
model.add(Dropout(0.2))

model.add(Dense(8,activation='relu',W_constraint=maxnorm(3)))
# The output layer
model.add(Dense(2, activation='softmax'))
#----------------------end of model---------------------
#compiling the model with loss parameter for binary image classificatoin and optimizer is selected to adam
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#checkpointing the model to save the model
filepath = "/home/shreyas0906/PycharmProjects/project-swarm/weights.best.h5"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc' , verbose=1, save_best_only=True,mode=max)
callbacks_list = [checkpoint]

#Training the model
model.fit(train_imgs, train_label, nb_epoch=epochs, batch_size=5,callbacks=callbacks_list,verbose=1)

#Testing the model
scores = model.evaluate(test_images, test_label,verbose=1)

#saving the model architecture into .json file for future use
model_json = model.to_json()
with open("model.json", 'w') as json_file:
    json_file.write(model_json)

#Saving the weights of the model
model.save_weights("model.h5")
print("model saved to disk")

#Displaying the network architecture
print(model.summary())
print("Accuracy: %.2f%%"%(scores[1]*100))

#Sample testing
test_img = cv2.imread('trn_174-roi.jpeg',cv2.IMREAD_GRAYSCALE)
test_img_2 = cv2.imread('trn_172-test.jpg',cv2.IMREAD_GRAYSCALE)

#reshaping the sample input image
img_1 = cv2.resize(test_img,(45,45))
img_2= cv2.resize(test_img_2,(45,45))

#reshaping the sample input to match the input of the trained model
img_1 = np.expand_dims(np.expand_dims(img_1, axis=0), axis=0)/255 #normalizing the sample input
img_2 = np.expand_dims(np.expand_dims(img_2, axis=0), axis=0)/255#normalizing the sample input

#Displaying the scores of the sample input
test_img = model.predict(img_1)
test_img_2 = model.predict(img_2)

print(test_img)
print(test_img_2)