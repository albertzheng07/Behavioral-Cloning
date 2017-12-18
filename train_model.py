import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

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
	measurements.append(float(line[3]))

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential() # build sequential model
model.add(Flatten(input_shape=(160,320,3))) # flatten input into single array
model.add(Dense(1)) # single dimensionality since input is flattened to N X 1 array

model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=7)
model.save('model_test')
