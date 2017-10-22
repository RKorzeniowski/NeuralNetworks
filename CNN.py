# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:54:19 2017

@author: buddy
"""

#Part 1 building CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialiasing the CNN
classifier = Sequential()

#better effects with bigger target size than (64,64) but use GPU to make it faster (here 1/2)
# setp 1 convolution. 1 layer (tensorflow has diffrent input than theano which has (RGB, size, size))
classifier.add(Convolution2D(32, 3, 3, border_mode = 'same', input_shape = (64, 64, 3), activation = 'relu'))

#step2 max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#second convolutional later (there is no input_shape becouse its not orginal image but already convoluted images goes inside and NN knows the size already)
#more Convolution2D(32... -> Convolution2D(64... feature maps improves it a lot
classifier.add(Convolution2D(32, 3, 3, border_mode = 'same', activation = 'relu'))
#maxpooing to the 2nd later
classifier.add(MaxPooling2D(pool_size=(2, 2)))

#step3 flattening
classifier.add(Flatten())

#step4 full connection layer 1 
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#step5 full connection layer 2 (not tested yet)
classifier.add(Dense(output_dim = 128, activation = 'relu'))

#output layer
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

#compiling the CNN (if more than binary like 3 loss = 'category_crossentropy')
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#fitting CNN to the image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


#better effects with bigger target size than (64,64) but use GPU to make it faster (here 2/2)
traning_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(traning_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000,
                    use_multiprocessing=True,
                    workers=6)
                    



import numpy as np
from keras.preprocessing import image
#load image to size
test_img = image.load_img('dataset/single_prediction/cato4.jpg',target_size=(64, 64))
#modifie pic to 3d array (R,G,B)
test_img = image.img_to_array(test_img)

#add dim to image 4rd dim is a number of badge (what, which is the new axis)
test_img = np.expand_dims(test_img, 0)

traning_set.class_indices
result = classifier.predict(test_img)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

