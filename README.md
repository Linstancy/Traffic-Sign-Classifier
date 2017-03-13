#**Traffic Sign Recognition using Deep Learning** 

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

[//]: # (Image References)

[image1]: ./analysis_images/bar_charts.png "Histograms"
[image2]: ./analysis_images/augmentation.png "Augmentation"
[image3]: ./analysis_images/confusion_matrix.png "Confusion Matrix on Validation Set"
[image4]: ./analysis_images/1.png "Traffic Sign 1"
[image5]: ./analysis_images/2.png "Traffic Sign 2"
[image6]: ./analysis_images/3.png "Traffic Sign 3"
[image7]: ./analysis_images/4.png "Traffic Sign 4"
[image8]: ./analysis_images/5.png "Traffic Sign 5"
[image9]: ./analysis_images/comparison.png "Comparison between two mis-predicted signs"

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

The code for this step is contained in the 6th and 7th code cell of the IPython notebook.

I generate one augmented image for every training image through randomly choosen translation, affine transform and rotation. All these operations are limited to certain scale in order to make sure a given traffic sign doesn't look like another sign after the transformations such as 90 degree or 180 degree rotations. The computational complexity (and hence time) and memory constrained me from generating more augmented data.

Here is an example of a traffic sign image and its corresponding randomly augmented image.

![alt text][image2]

As a next preprocessing step, I normalized the training image data because this ensures the features used to train the network are of the same scale(zero mean and unit variance) and hence the optimizer can use the same learning rate for all the weights and bias vectors being optimized to achieve minimum loss. I also pre-processed the validation and testing datasets by using the mean and standard deviations computed for the training set.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The data given for this project was already split into training, validation and testing datasets. 

The 6th code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data in order to make the model more robust to changes in real-world conditions such as viewing the sign from an angle, occlusions, brightness variation etc.. To add more data to the the data set, I defined a function which takes in an input image, applies brightness normalization and randomly applies translation (limited to 5 pixels), rotation (limited to +/- 10 degress) and limited affine transformation. I generated one augmented image per training image and consider them be a part of the new augmented training set. My final training set had 69598 number of images. My validation set and test set had 4410 and 12630 number of images respectively.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 8th cell of the ipython notebook. 

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

The code for training the model is located in the 9th and 10th cell of the ipython notebook. 

To train the model, I used a softmax with cross entropy loss and added L2 regularization. The regularizer term was added to the loss with a weight factor of 0.005(beta).  Adam Optimizer with exponetially decaying learning rate was used. The learning rate decays at the rate of 0.99 every epoch. This was found to be the ideal decay rate owing to the fact that the learning rate was still 80% of the original rate after 20 epochs and 60% after 50 epochs. The initial learning rate is set to 0.001 after trial and error. I employed a batch size of 64 which improved accuracy by 1-1.5% on validation set compared to other larger batch sizes. 

I also employed early termination. I saved the model whenever the validation accuracy improved compared to the previous best and stopped training when validation accuracy didn't increase for last 20 epochs.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 98.37% 
* test set accuracy of 97.17%

* I started with LeNet architecture which had two 5x5 convolutional layers followed by three fully connected layers. It yielded validation accuracy of 87% which I considered as baseline.It was a very simple architecture and was proven to work for digits classification task with 10 classes. Since traffic sign recognition problem had 43 classes and the task itself was more difficult compared to digit recogintion, the model wasn't able to capture all the necessary features with just two 5x5 convolutional layers. 

* The next iteration was to simply increase the number of feature maps at each convolutional layers and also add another fully connected layer to the network. Although this increased the validation accuracy to around 95%, the model was overfitting to the training data given the fact that the training error reached 99.5% within first five to ten epochs and eventually reached 100%, but validation accuracy increased sluggishly and never increased beyond 95%. This also made me reduce the learning rate and employ exponentially decaying learning rate. In order to minimize overfitting, I added L2 regularization for the loss and also added dropout to each stage except the first convolutional layer. 

* In order to make the network more deeper and wider to capture more features, I reduced the convolution filter sizes to 3x3 from 5x5 and increased the number of features maps to 16, 32 and 64 at each convolution layer respectively. I also added a third convolutional layer before the four fully connected layers. This increased the complexity of the network and with reduced batch size of 64 (from 128), the network now yielded 97.5% validation accuracy.
 
* I tried different color spaces (HSV, YUV) and found little improvement. After reading few online materials, I employed a  1x1 convolutional layer at the beginning of the network with 3 channels to automatically act as a colorspace transformer before feeding the data in to the rest of the network. 

* I then employed different data augmentation techniques and finally converged to a function which randomly applied less than 5 pixels of translation, +/- 10 degrees of rotation and limited shearing/angle of view changes. I limited data augmentation to just one augmented image per every image in training set due to memory and computational time constraints. This data augmentation and automatic colorspace transformer increased the accuracy by around 0.8% to 98.37% on validation set, which is my final validation accuracy.

Below is the image depicting confusion matrix on validation set: 

![alt text][image3]

As you can see above, the network confused sign 41 (End of no passing) as sign 32(End of all speed and passing limits) the most. Below are samples of images from each of those two classes which shows that they look very similar:

![alt text][image9]

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th, 13th and 14th cells of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h	      		| 30 km/h					 				|
| Stop Sign      		| Stop sign   									| 
| No Entry	| No Entry     							|
| Go straight or right     			| Go straight or right										|
| Right of way at the next intersection					| Right of way at the next intersection											|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.17%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for computing the top 5 softmax probabilities for each test image is located in the 15th cell of the Ipython notebook. All the traffic signs have large probabilities (>0.99) for their correct label ID. It is clear from the data below that the top 5 signs that get picked are mostly the ones which look similar to the input image. 

For example, 30 km/h speed limit sign has other speed limit signs in its top 5 predictions. "No Entry" sign has other signs which are mainly "no passing" type of signs. "Go straight and right" sign has similar signs such as Ahead only, turn right, keep left signs. The "Right of way at the next intersection" looks similar to "Pedestrians" sign and as expected, it is the second choice from the network. 
 
For the 30 km/h image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9926	      		| 30 km/h					 				|
| 0.0042	| 20 km/h     							|
| 0.0014     			| 50 km/h										|
| 0.0008     		|  70 km/h  									| 
| 0.0004				| Stop											|

For the stop sign image, the top five soft max probabilities were
 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9994	      		| Stop sign				 				|
| 0.0002	| No Entry     							|
| 0.0001     			| Traffic signals									|
| 0.0001     		| Priority road   									| 
| 0.0001				| Bicycles crossing											|

For the "No Entry" image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9997	      		| No Entry					 				|
| 0.0003	| Stop     							|
| 0.0000     			| Priority road									|
| 0.0000     		| No passing for vehicles over 3.5 metric tons   									| 
| 0.0000				| No passing											|

For the "Go straight or right" image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9984	      		| Go straight or right					 				|
| 0.0007	| Ahead only     							|
| 0.0004     			| Keep right										|
| 0.0004     		| Turn right ahead  									| 
| 0.0001				| Keep Left											|


For the "Right of way at the next intersection" image, the top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9934	      		|  Right of way at the next intersection				 				|
| 0.0054	| Pedestrians    							|
| 0.0010     			| Beware of ice or snow										|
| 0.0001     		| Double curve   									| 
| 0.0000				| Children crossing											|
