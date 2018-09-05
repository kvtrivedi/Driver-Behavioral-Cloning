import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import csv

lines = [] 
#extrating each line from csv file to obtain the image files
with open('./data/data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None) 
    for line in reader:
        lines.append(line)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#splitting the test and validation data
train_lines, validation_lines = train_test_split(lines,test_size=0.15) 


import cv2
import sklearn
import matplotlib.pyplot as plt

#Using a generator
def generator(lines, batch_size=32):
    num_lines = len(lines)
   
    while 1:
        #to randomly obtain data each time
        shuffle(lines) 
        for offset in range(0, num_lines, batch_size):
            
            batch_lines = lines[offset:offset+batch_size]

            images = []
            steering_angles = []
            for line in batch_lines:
                    for i in range(0,3):
                        #loop iterations are 3 for centre, left and right images
                        name = './data/data/IMG/'+line[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) 
                        center_angle = float(line[3]) 
                        images.append(center_image)
                        #correcting the steering angle in case of left or right images
                        if(i==0):
                            steering_angles.append(center_angle)
                        elif(i==1):
                            steering_angles.append(center_angle+0.2)
                        elif(i==2):
                            steering_angles.append(center_angle-0.2)
                        
                        #Negated steering angles for augmented data that represents
                        #movement in the counter clockwise direction
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            steering_angles.append(center_angle*-1)
                        elif(i==1):
                            steering_angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            steering_angles.append((center_angle-0.2)*-1)
                               
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) 
            

train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)

#Defining the model
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

#Preprocessing images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#Crop out unwanted portions of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))           

#Layer 1 - Convolution
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#Layer 2 - Convolution
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#Layer 3 - Convolution
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))

#Layer 4 - Convolution
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#Layer 5 - Convolution
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))

#Flattening
model.add(Flatten())

#Fully connected layer 1
model.add(Dense(100))
model.add(Activation('elu'))

#Dropout
model.add(Dropout(0.25))

#Fully connected layer 2
model.add(Dense(50))
model.add(Activation('elu'))


#Fully connected layer 3
model.add(Dense(10))
model.add(Activation('elu'))

#Fully connected layer 4
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch= len(train_lines), validation_data=validation_generator,   nb_val_samples=len(validation_lines), nb_epoch=5, verbose=1)

model.save('model.h5')

model.summary()

