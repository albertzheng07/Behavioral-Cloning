import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
# open up csv file which contains driving logs
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader: 
		lines.append(line) # append in each line of csv data into list

images = []
measurements = []

for line in lines[1:]:
	source_path = line[0]
	filename = source_path.split('/')[-1] # get filename from path
	curr_path = '../data/IMG/' + filename
	image = cv2.imread(curr_path)
	images.append(image)
	image_flipped = np.fliplr(image)
	images.append(image_flipped)
	measurement = float(line[3])	
	measurement_flipped = -measurement	
	measurements.append(measurement)
	measurements.append(measurement_flipped)	

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential() # build sequential model
model.add(Lambda(lambda x: x/255-0.5,input_shape=(160,320,3)))
model.add(Convolution2D(15,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(15,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten()) # flatten input into single array
model.add(Dense(120)) 
model.add(Dense(84)) 
model.add(Dense(1)) # single dimensionality since input is flattened to N X 1 array

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=10)
model.save('model_test')
exit()
