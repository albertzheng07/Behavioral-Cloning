import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#import ipdb

lines = []
# open up csv file which contains driving logs
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader: 
		lines.append(line) # append in each line of csv data into list

images = []
measurements = []

use_NN_arch = 1 # set 0 for LeNet baseline, set 1 for Nvidia

def get_image_path(source_path):
	filename = source_path.split('/')[-1] # get filename from path
	curr_path = '../data/IMG/' + filename
	return curr_path

correction_factor = [0.0,-0.3,0.3]

for line in lines[1:2]:
	for ind in range(3):
		img_path = get_image_path(line[ind])
		image = cv2.imread(img_path)
		images.append(image)
		image_flipped = np.fliplr(image)
		images.append(image_flipped)
		steering = float(line[3])+correction_factor[ind] # steering measurement
		steering_flipped = -steering	
		measurements.append(steering)
		measurements.append(steering_flipped)	


X_train = np.array(images)
y_train = np.array(measurements)

if use_NN_arch == 0:
	model = Sequential() # build sequential model of LeNet
	model.add(Lambda(lambda x: x/255-0.5,input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop out the top 75 pixels, bottom 25 pixels, none n the outside
	model.add(Convolution2D(15,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Convolution2D(15,5,5,activation="relu"))
	model.add(MaxPooling2D())
	model.add(Flatten()) # flatten input into single array
	model.add(Dense(120)) 
	model.add(Dense(84)) 
	model.add(Dense(1)) # single dimensionality since input is flattened to N X 1 array
elif(use_NN_arch == 1):
	model = Sequential() # build sequential model of LeNet
	model.add(Lambda(lambda x: x/255-0.5,input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop out the top 75 pixels, bottom 25 pixels, none n the outside
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten()) # flatten input into single array
	model.add(Dense(100)) 
	model.add(Dense(50)) 
	model.add(Dense(10)) 

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)


model.save('model_test')
exit()
