# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model1]: ./my_results/model.png "Model Visualization"
[model2]: ./my_results/model_last.png "Model summary"
[image2]: ./my_results/center.jpg "Recovery Image"
[image3]: ./my_results/left.jpg "Recovery Image"
[image4]: ./my_results/right.jpg "Recovery Image"
[image5]: ./my_results/center_flip.jpg "Flipped Image"
[image6]: ./my_results/left_flip.jpg "Flipped Image"
[image7]: ./my_results/right_flip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 97-117) 

The model includes RELU layers to introduce nonlinearity (code line 110 and between 108 and 130 layers), and the data is normalized in the model using a Keras lambda layer (code line 109). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layer with 0.5 coefficient in order to reduce overfitting (model.py line 108). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 91). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 114).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, brightness augmentation and several augmentation strategies so it is driving nicely.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to drive safely so I limited the speed to 9km.

My first step was to use a convolution neural network model similar to the Comma AI's model. I thought this model might be appropriate because it is modifiable, starts with 5x5 filters with filter number of 16 and it was using ELU. By increasing filter number, decreasing filter size and adding relu I could make more accurate model for behavioral cloning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model with dropout layers so it reaches the optimal solution faster, and without overfitting.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 97-117) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][model1]
![alt text][model2]
#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:
![alt text][image2]


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive autonomously. These images show what a recorded images looks like starting from left to right :
But at the end I used Udacity dataset.

![alt text][image2]
![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would allow to increase images. For example, here is set of images that has then been flipped:
![alt text][image5]
![alt text][image6]
![alt text][image7]

After the collection process, I had increased the number of data points. I then preprocessed this data by random brightness.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by stable loss results. I used an adam optimizer, so that manually training the learning rate wasn't necessary.
