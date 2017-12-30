# **Behavioral Cloning**

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./media/loss.png "Model Loss"
[image2]: ./media/accuracy.png "Model Accuracy"
[image3]: ./media/center_lane.jpg "Center lane"
[image4]: ./media/recover1.jpg "Recovery Image"
[image5]: ./media/recover2.jpg "Recovery Image"
[image6]: ./media/recover3.jpg "Recovery Image"
[image7]: ./media/center_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](./model.py) containing the script to create and train the model
* [drive.py](./drive.py) for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network*
* [video.mp4](./media/video.mp4)
* this readme file summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 82-99).

* The first layer of the model performs normalization using a Keras lambda layer (code line 84).
* The image is then cropped to focus on the road features (line 86).
* The following three convolutional layers use strided convolutions (lines 88-90) and the last two layers are non-strided. All layers use RELU to introduce nonlinearity.
* Convolutional layers are followed by 4 fully connected layers (lines 95-98), terminating in a single value representing the turning angle.

#### 2. Attempts to reduce overfitting in the model

In order to generalize the model the training data set was mirrored (lines 60-66), which helped ensure the model did not memorize the left-hand biased track. The car was driven in both direction of the track to generalize the training data.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (7 laps), recovering from the left and right sides of the road (3 laps forward, 1 in reverse), and driving in the opposite direction of the track to remove the left-hand bias (3 laps).

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to leverage a convolutional network to recognize image features, followed by a neural network to learn driving behavior as it relates to these features and the human driver choices when steering the car.

My first step was to use a convolutional neural network model similar to the [Nvidia self-driving car model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). I thought this model might be appropriate because it was derived from a series of real-life experiments.

In order to evaluate the model, I split my image and steering angle data into a training and validation set (80/20). My first model relied on training data from 4 laps and it did not perform well on the track.

I recorded additional driving data (10 laps total). This made immediate difference in the car's performance and it was able to complete the loop by using only center camera.

The car was still going off the road in a few places. I recorded 4 laps of driving the car off the side of the road, which addressed this problem. I extended the training data by using left and right cameras.

The model was still overfitting, which was apparent from the low mean squared error on the training set but a high mean squared error on the validation set. I added three three dropout layers (50%) in-between the connected layers, which helped the model to converge better

![alt text][image1]

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road at a speed of 30 mph. The driving was smooth and better than my manual driving.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I recorded 10 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center when running off the lane. These images show what a recovery looks like:

![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data set, I added images with varying brightness, added random shadows, and flipped images and angles to generalize the model away from left turns. For example, here is an image that has been flipped:

![alt text][image7]

After the collection process, I had 38,130 data points. I doubled the amount of data by adding mirror images and added 80% more images by augmenting random images (brightness, shadows). I then preprocessed this data by cropping the image to focus training on the features close to the road.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4-5 as evidenced by the point of convergence between the training and validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary. I also discovered that using smaller batch sizes (32) made the model converge better than using large batch sizes (312).
