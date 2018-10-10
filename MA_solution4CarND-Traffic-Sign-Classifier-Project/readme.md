# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/training_set_counts.png "Counts"
[image2]: ./images/lenet-to-resnet-17-638.jpg "ResNet"
[image3]:./images/resmeacc.png "Loss"
[image4]: ./images/resmeloo.png "Accuracy"
[image5]: ./images/0000005.jpeg "Sign 1"
[image6]: ./images/00000012.jpeg "Sign 2"
[image7]: ./images/0000027.jpeg "Sign 3"
[image8]: ./images/00000028.jpeg "Sign 4"
[image9]: ./images/00000042.jpeg "Sign 5"
[image10]: ./images/intersection.png
[image11]: ./images/slippery.png
[image12]: ./images/stop-no-park.png
[image13]: ./images/27.png
[image14]: ./images/lights.png


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/meltem-ai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier-Keras_models_final.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Original LeNet version on traffic sign dataset, namely SermaNet was using YUV space, grascaled images. 
I believe that CNNs use the adventage of colors so I did not grayscaled my images, nor changed the colorspace, instead I performed preprocessing using Keras pipeline. Keras preprocessing is fairly easy to implement. 
I normalized and shuffled dataset, and performed preprocessing with following steps, bright enhancement, contrast enhancement, rotation to right by 15 degrees, shearing, width and height shifting, horizontal and vertical flips. By this type of processing I enhanced the learning ability of my models. To prove this claim I used LeNet implementation on Tensorflow, and implemented LeNet using Keras, also used a deeper CNN and finally ResNet20 using Keras.
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
My best performing model is ResNET20v1

x_train shape: (34799, 32, 32, 3)
34799 train samples
12630 test samples
y_train shape: (34799,)
Learning rate:  0.001

|Layer (type)          |          Output Shape     |    Param #   |  Connected to                 |    
|----------------------|:--------------------------:|------------:|------------------------------:|
|input_1 (InputLayer)  | (None, 32, 32, 3) |   0         |                               |   
|conv2d_1 (Conv2D)     |(None, 32, 32, 16) |   448      |   input_1[0][0]               |     
|batch_normalization_1(BatchNor |(None, 32, 32, 16)   |64          |conv2d_1[0][0]                   |
|activation_1 (Activation) | (None, 32, 32, 16)   |0           |batch_normalization_1[0][0]      |
|conv2d_2 (Conv2D)     | (None, 32, 32, 16)   |2320        |activation_1[0][0]               
|batch_normalization_2 (BatchNor |(None, 32, 32, 16)   |64          |conv2d_2[0][0]                   |
|activation_2 (Activation) |  (None, 32, 32, 16)   |0           |batch_normalization_2[0][0]      |
|conv2d_3 (Conv2D)      |(None, 32, 32, 16)   |2320        |activation_2[0][0]               |
|batch_normalization_3 (BatchNor |(None, 32, 32, 16)   |64         | conv2d_3[0][0]                   |
|add_1 (Add)            |         (None, 32, 32, 16)   |0           |activation_1[0][0]  batch_normalization_3[0][0]  |          
|activation_3 (Activation)| (None, 32, 32, 16)| 0           |add_1[0][0]                      |
|conv2d_4 (Conv2D)        | (None, 32, 32, 16) |2320        |activation_3[0][0]               |
|batch_normalization_4    (BatchNor|(None, 32, 32, 16)   |64          |conv2d_4[0][0]                   |
|activation_4 (Activation)| (None, 32, 32, 16)   |0           |batch_normalization_4[0][0]      |
|conv2d_5 (Conv2D)        | (None, 32, 32, 16)   |2320        |activation_4[0][0]               |
|batch_normalization_5 (BatchNor |(None, 32, 32, 16)   |64          |conv2d_5[0][0]                   |
|add_2 (Add)            |       (None, 32, 32, 16)  | 0           |activation_3[0][0] batch_normalization_5[0][0]               |
|activation_5 (Activation)       | (None, 32, 32, 16)   |0           |add_2[0][0]                    |  
|conv2d_6 (Conv2D)               |  (None, 32, 32, 16)   |2320        |activation_5[0][0]             |  
|batch_normalization_6 (BatchNor | (None, 32, 32, 16)   |64          conv2d_6[0][0]                   |
|activation_6 (Activation)       | (None, 32, 32, 16)   |0           batch_normalization_6[0][0]      |
|conv2d_7 (Conv2D)               | (None, 32, 32, 16)   |2320        activation_6[0][0]               |
|batch_normalization_7 (BatchNor | (None, 32, 32, 16)   |64          conv2d_7[0][0]                   |
|add_3 (Add)                     | (None, 32, 32, 16)   |0           activation_5[0][0]    batch_normalization_7[0][0]     |
|activation_7 (Activation)       | (None, 32, 32, 16)   |0           |add_3[0][0]     |                 
|conv2d_8 (Conv2D)              |  (None, 16, 16, 32)   |4640        |activation_7[0][0]         |      
|batch_normalization_8 (BatchNor | (None, 16, 16, 32)   |128         |conv2d_8[0][0]             |      
|activation_8 (Activation)       | (None, 16, 16, 32)  | 0           |batch_normalization_8[0][0]|      
|conv2d_9 (Conv2D)               | (None, 16, 16, 32)  | 9248        |activation_8[0][0]         |      
|conv2d_10 (Conv2D)              | (None, 16, 16, 32)   |544         |activation_7[0][0]          |     
|batch_normalization_9 (BatchNor |  (None, 16, 16, 32)  | 128         |conv2d_9[0][0]              |     
|add_4 (Add)                     | (None, 16, 16, 32)   |0           |conv2d_10[0][0]   batch_normalization_9[0][0]    |  
|activation_9 (Activation)       | (None, 16, 16, 32)   |0           |dd_4[0][0]   |              |     
|conv2d_11 (Conv2D)              | (None, 16, 16, 32)   |9248        |activation_9[0][0]             |  
|batch_normalization_10 (BatchNo | (None, 16, 16, 32)   |128         |conv2d_11[0][0]                |  
|activation_10 (Activation)      | (None, 16, 16, 32)  | 0           |batch_normalization_10[0][0]   |  
|conv2d_12 (Conv2D)              | (None, 16, 16, 32)   |9248        |activation_10[0][0]            |  
|batch_normalization_11 (BatchNo | (None, 16, 16, 32)   |128         |conv2d_12[0][0]                |  
|add_5 (Add)                     | (None, 16, 16, 32)  | 0          | activation_9[0][0]  batch_normalization_11[0][0]   |  
|activation_11 (Activation)      | (None, 16, 16, 32)   |0           ||add_5[0][0]                    |  
|conv2d_13 (Conv2D)               | (None, 16, 16, 32)   |9248        activation_11[0][0]             | 
|batch_normalization_12 (BatchNo | (None, 16, 16, 32)   |128         |conv2d_13[0][0]                 | 
|activation_12 (Activation)      | (None, 16, 16, 32)   |0          | batch_normalization_12[0][0]    | 
|conv2d_14 (Conv2D)              | (None, 16, 16, 32)   |9248        |activation_12[0][0]              |
|batch_normalization_13 (BatchNo | (None, 16, 16, 32)   |128         |conv2d_14[0][0]                 | 
add_6 (Add)                     | (None, 16, 16, 32)   |0           |activation_11[0][0]  batch_normalization_13[0][0]    | 
|activation_13 (Activation)      | (None, 16, 16, 32)  | 0           |add_6[0][0]                    |  
|conv2d_15 (Conv2D)              | (None, 8, 8, 64)     |18496      | activation_13[0][0]            |  
|batch_normalization_14 (BatchNo |  (None, 8, 8, 64)     256         |conv2d_15[0][0]                  
|activation_14 (Activation)      |(None, 8, 8, 64)    | 0           |batch_normalization_14[0][0]  |   
|conv2d_16 (Conv2D)              |  (None, 8, 8, 64)     |36928       |activation_14[0][0]             | 
|conv2d_17 (Conv2D)              | (None, 8, 8, 64)     |2112        |activation_13[0][0]            |  
|batch_normalization_15 (BatchNo | (None, 8, 8, 64)    | 256         |conv2d_16[0][0]                |  
|add_7 (Add)                     | (None, 8, 8, 64)     |0           |conv2d_17[0][0]     batch_normalization_15[0][0]     |
|activation_15 (Activation)      | ( None, 8, 8, 64)     |0          | add_7[0][0]                      |
|conv2d_18 (Conv2D)              | (None, 8, 8, 64)     |36928      | activation_15[0][0]              |
|batch_normalization_16 (BatchNo | (None, 8, 8, 64)     |256         |conv2d_18[0][0]                 | 
|activation_16 (Activation)      | (None, 8, 8, 64)    | 0          |batch_normalization_16[0][0]    | 
|conv2d_19 (Conv2D)              | (None, 8, 8, 64)     |36928      | activation_16[0][0]            |  
|batch_normalization_17 (BatchNo | (None, 8, 8, 64)     |256        | conv2d_19[0][0]                |  
|add_8 (Add)                     | (None, 8, 8, 64)     |0         |  activation_15[0][0]  batch_normalization_17[0][0]   |  
|activation_17 (Activation)      | (None, 8, 8, 64) |    0         |  add_8[0][0]                 |     
|conv2d_20 (Conv2D)              | (None, 8, 8, 64) |    36928    |   activation_17[0][0]          |    
|batch_normalization_18(BatchNo  |(None, 8, 8, 64)  |   256      |   conv2d_20[0][0]              |    
|activation_18 (Activation)      | (None, 8, 8, 64)  |   0        |   batch_normalization_18[0][0]  |   
|conv2d_21 (Conv2D)              | (None, 8, 8, 64)  |   36928    |   activation_18[0][0]          |    
|batch_normalization_19 (BatchNo | (None, 8, 8, 64)  |   256      |   conv2d_21[0][0]              |    
|add_9 (Add)                    | (None, 8, 8, 64)  |   0        |   activation_17[0][0]    batch_normalization_19[0][0]  |   
|activation_19 (Activation)      | (None, 8, 8, 64)  |   0        |   add_9[0][0]      |                
|average_pooling2d_1| (AveragePoo  |(None, 1, 1, 64) |    0        |   activation_19[0][0]|              
|flatten_1 (Flatten)             | (None, 64)       |    0       |    average_pooling2d_1[0][0] |       
|dense_1 (Dense)                 | (None, 43)        |   2795   |     flatten_1[0][0]        |          


Total params: 276,587
Trainable params: 275,211
Non-trainable params: 1,376
ResNet20v1
![alt text][image2]


This picture also describes that ResNet allows more dense and deeper CNNs to enhance accuracy because of the skip connections
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For all of my models except 4 layer CNN, I used Adam optimizer, for 4 layer CNN, I used RMSprop. Batch size is 128, and trained all of my models at least 150 epochs, but for my best performing ResNet20 model I trained for 200 epochs. For the best performing model, I started with learning rate of 0.1 and as epochs increased with assumption of reaching the global minima, I decreased the learning rates to 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My best performing model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.9912
* test set accuracy of 9784

![alt text][image3]


![alt text][image4]


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I used LeNEt implementation because it is mentioned in the lecture.
* What were some problems with the initial architecture?
LeNet is too basic for multiclass classification.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
There were some underfitting at first epochs then overfitting, I used dropout and changed the learning rate. But these approaches did not increase the accuracy so I changed the model.
* Which parameters were tuned? How were they adjusted and why?
Added dropout and made earning rate bigger to train faster.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The problem is object classification, so convolutional layers are providing best filter solutions for this problem and dropout helps to overcome overfitting by resulting more robust outputs.

If a well known architecture was chosen:
* What architecture was chosen?
I chose ResNet with 20 layers.
* Why did you believe it would be relevant to the traffic sign application?
ResNet contains skip connections.
[This is ResNet image][image2] 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 After first 100 epochs model reaches to plateau of 1.000 training accuracy, this means model is training very well, for validation model shows some jittering but it stays consistent, and correlates with test accuracy, this measn ResNet20 is a good model for this problem.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.
Instead of selecting images from the internet I went outside and took some photos.


![alt text][image5] ![alt text][image6] ![alt text][image7] 
![alt text][image8] ![alt text][image9]

The first image indicates to be slower than 20km/h
Second image is he General Caution Sign
Third image is Children Crossing, this image might not be unique all around the word so this may be confusing.
Fourth image is 30km/h speed limit , due to high daylight this may look naturally adversarial
Fifth image is Traffic signals

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop-No parking     		| Stop  									| 
| Right-of-way at the next intersection | Right-of-way at the next intersection|
| Children Crossing				| Turn left ahead											|
| Slippery road|Slippery road|
| Traffic Signals		| Traffic Signals      							|


The LeNEt model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This is less than training set accuracy, but my images are more challenging than the dataset and I did not preprocess them to show the effects of preprocessing before testing with the  model.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)
For all the images, even if the model makes absurd mistakes, it is undoubtedly sure about its prediction, this sort of mistake on LeNet model can be the underlying motivation on using deeper models.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop    									| 
| 1.0     				| Right-of-way 										|
| 1.0					| Turn left ahead											|
|1.0      			| Slippery road					 				|
| 1.0				    | Traffic Signals     							|


![alt text][image10] ![alt text][image11] ![alt text][image12] 
![alt text][image13] ![alt text][image14]
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


