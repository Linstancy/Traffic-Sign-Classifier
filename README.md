#**Traffic Sign Recognition using Deep Learning** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the README file ! and here is a link to my [project code](https://github.com/rajkn90/Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of test set is 12630 images
* The shape of a traffic sign image is (32, 32, 3) unsigned integer numpy array
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third, fourth and fifth code cell of the IPython notebook.  

One image per class is displayed as well as four bar charts showing how the histogram of labels in the training, validation and test set individually and also in the overall dataset. Different traffic signs occur at different frequencies and in real world, this is mostly the case since some traffic signs are displayed more than the other in any given city or country. This prior distribution of labels is unchanged while training the network owing to the fact that we want our neural network to know this real-world distribution and learn to classify more frequent traffic signs better than the others. 

Here are the bar charts:

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth and sixth code cell of the IPython notebook.

I generate one augmented image for every training image through randomly choosen translation, affine transform and rotation. All these operations are limited to certain scale in order to make sure a given traffic sign doesn't look like another sign after the transformations such as 90 degree or 180 degree rotations. The computational complexity (and hence time) and memory constrained me from generating more augmented data.

Here is an example of a traffic sign image and its corresponding randomly augmented image.

![alt text][image2]

As a next preprocessing step, I normalized the training image data because this ensures the features used to train the network are of the same scale(zero mean and unit variance) and hence the optimizer can use the same learning rate for all the weights and bias vectors being optimized to achieve minimum loss. I also pre-processed the validation and testing datasets by using the mean and standard deviations computed for the training set.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data given for this project was already split into training, validation and testing datasets. 

The fifth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data in order to make the model more robust to changes in real-world conditions such as viewing the sign from an angle, occlusions, brightness variation etc.. To add more data to the the data set, I defined a function which takes in an input image, applies brightness normalization and randomly applies translation (limited to 5 pixels), rotation (limited to +/- 10 degress) and limited affine transformation. I generated one augmented image per training image and consider them be a part of the new augmented training set. My final training set had 69598 number of images. My validation set and test set had 4410 and 12630 number of images respectively.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x3 (Learns best color space)	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x32	|
| RELU					|												|
| Dropout					|				With keep probability = 0.5								|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x64	|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride,  outputs 14x14x64 				|
| Dropout					|				With keep probability = 0.5								|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x128     									|
| RELU					|												|
| Max pooling	2x2      	| 2x2 stride,  outputs 6x6x128 				|
| Dropout					|				With keep probability = 0.5								|
| Fully connected		| Inputs = 4608 (6x6x128)  Outputs = 1024       									|
| RELU					|												|
| Dropout					|				With keep probability = 0.5								|
| Fully connected		| Inputs = 1024 Outputs = 1024       									|
| RELU					|												|
| Dropout					|				With keep probability = 0.5								|
| Fully connected		| Inputs = 1024 Outputs = 512       									|
| RELU					|												|
| Dropout					|				With keep probability = 0.5								|
| Fully connected		| Inputs = 512 Outputs = 43 (Number of classes)       									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 
####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used a softmax with cross entropy with L2 regularization as the loss. The regularizer term was added to the loss with a weight factor of 0.005.  Adam Optimizer with exponetially decaying learning rate was used. The learning rate decays at the rate of 0.99 every epoch. This was found to be the ideal decay rate owing to the fact that the learning rate was still 80% of the original rate after 20 epochs and 60% after 50 epochs. The initial learning rate is set to 0.001 after trial and error. I employed a batch size of 32 which improved accuracy by 1-1.5% on validation set compared to other larger batch sizes. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.7% accuracy
* validation set accuracy of 98.0% accuracy
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
