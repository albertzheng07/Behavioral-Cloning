# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/2017_12_22_20_54_08_104.jpg "Center Lane Driving"
[image3]: ./examples/center_2017_12_22_19_07_01_251.jpg "Recovery Image 1"
[image4]: ./examples/center_2017_12_22_19_07_01_613.jpg "Recovery Image 2"
[image5]: ./examples/center_2017_12_22_19_07_02_059.jpg "Recovery Image 3"
[image6]: ./examples/baseline_img.jpg "Baseline Image"
[image7]: ./examples/flipped_img.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train_model.py containing the script to create and train the neural network model
* drive.py for driving the car in autonomous mode
* model_track1 contains a trained convolution neural network which drives autonomously on the easier track 1
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model_track1
```

#### 3. Submission code is usable and readable

The model_train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

A deep convolutional neural network has been implemented using Keras which mimics the Nvidia deep neural network. It employs 5 convolutional 2 d layers with 4 different filters sizes.

The convolution layers are in sequential order and did not require drop out layers in between each for the final result. The ordering and size of the convolutional layers can be seen in Lines 61-65 as seen below.

```
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
```

#### 3. Model parameter tuning

The model used an adam optimizer which was selected to minimize the loss function of the mean squared error. I selected 3 epochs as the N times required to propagate back and forth through the model in order to maximize the model training without diminishing returns.

#### 4. Appropriate training data

I gathered training data that was designed to keep the vehicle on the road during the entire course. I used multiple collection strategies in order to provide enough data for the network to be robust to various turns and road changes throughout the course.
The following strategies were used:
1. Baseline center of the road tracking
2. Recovery back to the center of the road when driving towards the left or right side
3. Driving the course in the counter-clockwise direction
4. Using multiple camera images from different views
5. Flipping the images to augment the data set
6. Bias the steering magnitude with the corresponding view

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was as follows:

1. The first step was to use a convolution neural network model which is based on LeNet. Since Lenet has proven well in the past on classifying image data, I thought it was an appropriate baseline model.

2. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I then evaluated how well the model was minimizing the loss function output from the fitting.

3. The next step was to run the simulator to see how well the car was driving around track one. I found quickly that the car quickly diverged off the track.

4. I then attempted to increase the data augmentation by flipping images, adding more images at different view points and providing more data sets where the vehicle would correct back to center when drifting away the road.

5. I found that LeNet was not successful in completing the autonomous lap even when providing the model with augmented data. I decided next to attempt a more powerful
network with more convolutional layers.

6. I selected the Nvidia neural network as the deeper neural net Architecture
since this was a proven model with success on actual vehicles.

7. In the end, I found that the more powerful network was able drive the lap on the same dataset and decided to use the Nvidia architecture as my design architecture.

#### 2. Final Model Architecture

The final model architecture (train_model.py lines 70-82) consisted of a convolution neural network.

```
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
```

My final model is based on the Nvidia Architecture which consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB  image   							|
| Normalization     | Normalize between [-0.5,0.5]							|
| Cropping     | Crop out Top 75 pixels and Bottom 25 pixels, 65 x 320 x 3 RGB image	|
| Convolution 5x5     	| 24 filters, 5x5 stride, outputs 31158x24 |
| RELU					|												|
| Convolution 5x5     	| 36 filters, 5x5 stride, outputs 14x77x36 |
| RELU					|												|
| Convolution 5x5     	| 48 filters, 5x5 stride, outputs 5x37x48 |
| RELU					|												|
| Convolution 5x5     	| 64 filters, 5x5 stride, outputs 3x35x64 |
| RELU					|												|
| Convolution 5x5     	| 64 filters, 5x5 stride, outputs 1x33x64 |
| RELU					|												|
| Flatten				|     Output 2112   									|
|	Fully Connected					|		Output 100										|
|	Fully Connected					|					Output 50							|
|	Fully Connected					|					Output 10							|
|	Fully Connected					|					Output 1						|

#### 3. Creation of the Training Set & Training Process

In order to capture good driving behavior, I first recorded two laps on track one using nominal center lane driving where I would not deviate from the center of the road. I did recorded each lap in both the clockwise and counter clockwise directions. Here is an example image of center lane driving:

![alt text][image2]

My next step was to then record the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself if there was deviation from the center of the road. These images show what a recovery will look like when heading towards the side of the road:

![alt text][image3]
![alt text][image4]
![alt text][image5]

In order to augment the data set, I also flipped images thinking that this would provide additional angles for the network to train on. Here is an example of a baseline images vs. a flipped image.

Baseline:
![alt text][image6]

Flipped:
![alt text][image7]

After the collection process, I had 5097 number of data points. I then preprocessed this data by adding the left and right images, and flipping all images. This resulted in a total of 30582 images.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the loss function.
