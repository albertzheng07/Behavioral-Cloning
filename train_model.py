import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

lines = []
# open up csv file which contains driving logs
with open('../test_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader: 
		lines.append(line) # append in each line of csv data into list

images = []
measurements = []

use_NN_arch = 1 # set 0 for LeNet baseline, set 1 for Nvidia
print_ex = False
print_shapes = False
save_model = True
run_model = True

def get_image_path(source_path):
	filename = source_path.split('/')[-1] # get filename from path
	curr_path = '../test_data/IMG/' + filename
	return curr_path

correction_factor = [0.0,0.2,-0.2]

n_lines = len(lines)

for line in lines[1:n_lines]:
	for ind in range(3):
		img_path = get_image_path(line[ind])
		image = cv2.imread(img_path)
		if print_ex:
			plt.imshow(image)
			plt.savefig('examples/baseline_img.jpg')
		images.append(image)
		image_flipped = np.fliplr(image)
		if print_ex:
			plt.imshow(image_flipped)
			plt.savefig('examples/flipped_img.jpg')		
		images.append(image_flipped)
		steering = float(line[3])+correction_factor[ind] # steering measurement
		steering_flipped = -steering	
		measurements.append(steering)
		measurements.append(steering_flipped)	

X_train = np.array(images)
y_train = np.array(measurements)

if run_model:
	# Setup Kera Neural Nets
	if use_NN_arch == 0:
		model = Sequential() # build sequential model of LeNet
		model.add(Lambda(lambda x: x/255-0.5,input_shape=(160,320,3)))
		model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop out the top 75 pixels, bottom 25 pixels, none n the outside
		model.add(Convolution2D(15,5,5,activation="relu")) # 1st conv layer of 15 filters, 5x5 stride, relu activation
		model.add(MaxPooling2D()) # max pooling layer 
		model.add(Convolution2D(15,5,5,activation="relu")) # 2nd conv layer of 15 filters, 5x5 stride, relu activation
		model.add(MaxPooling2D()) # max pooling layer 
		model.add(Flatten()) # flatten input into single array
		model.add(Dense(120)) # fully connected layer with output of 120
		model.add(Dense(84)) # fully connected layer with output of 84
		model.add(Dense(1)) # single dimensionality since input is flattened to N X 1 array
	elif(use_NN_arch == 1):
		model = Sequential() # build sequential model of Nvidia NN
		model.add(Lambda(lambda x: x/255-0.5,input_shape=(160,320,3)))
		model.add(Cropping2D(cropping=((70,25),(0,0)))) # crop out the top 75 pixels, bottom 25 pixels, none n the outside
		model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu")) # 1st conv layer of 24 filters, 5x5 stride, relu activation
		model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu")) # 2nd conv layer of 36 filters, 5x5 stride, relu activation
		model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu")) # 3rd conv layer of 48 filters, 5x5 stride, relu activation
		model.add(Convolution2D(64,3,3,activation="relu")) # 4th conv layer of 64 filters, 3x3 stride, relu activation
		model.add(Convolution2D(64,3,3,activation="relu")) # 5th conv layer of 64 filters, 3x3 stride, relu activation
		model.add(Flatten()) # flatten input into single array
		model.add(Dense(100)) # fully connected layer with output of 100
		model.add(Dense(50))  # fully connected layer with output of 50
		model.add(Dense(10))  # fully connected layer with output of 10
		model.add(Dense(1))	 # fully connected layer with output of 1

	# Print output layer shapes
	if print_shapes:
	print("Output Layer Shapes \n")
	outputs = [layer.output_shape for layer in model.layers]
	for ind, out in enumerate(outputs):
		print("layer %d",ind)
		print("shape = \n", out) 

	model.compile(loss='mse',optimizer='adam') # set mse loss function with adam optimizer
	model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3) # call fit function with training data and use 20% of data as validation data, shuffle data and run n epochs

if save_model:
	model.save('model_track1')
exit()
