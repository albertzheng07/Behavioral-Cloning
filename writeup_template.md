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
[image2]: ./examples/placeholder.png "Center Lane Driving"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
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
* writeup_report.md or writeup_report.pdf summarizing the results

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

The overall strategy for deriving a model architecture was to ...

1. My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

2.

3.

4.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

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
| Cropping     | Crop out Top 75 pixels and Bottom 25 pixels, 60 x 320 x 3 RGB image	|
| Convolution 5x5     	| 24 filters, 5x5 stride, outputs NxNxN |
| RELU					|												|
| Convolution 5x5     	| 36 filters, 5x5 stride, outputs NxNxN |
| RELU					|												|
| Convolution 5x5     	| 48 filters, 5x5 stride, outputs NxNxN |
| RELU					|												|
| Convolution 5x5     	| 64 filters, 5x5 stride, outputs NxNxN |
| RELU					|												|
| Convolution 5x5     	| 64 filters, 5x5 stride, outputs NxNxN |
| RELU					|												|
| Flatten				|     Output XXXX   									|
|	Fully Connected					|		Output 100										|
|	Fully Connected					|					Output 50							|
|	Fully Connected					|					Output 10							|
|	Fully Connected					|					Output 1						|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

In order to augment the data set, I also flipped images thinking that this would provide additional angles for the network to train on. Here is an example of a baseline images vs. a flipped image.

Baseline:
![alt text][image6]

Flipped:
![alt text][image7]


After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the loss function. I used an adam optimizer so that manually training the learning rate wasn't necessary.
