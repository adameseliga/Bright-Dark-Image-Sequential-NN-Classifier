# Bright-Dark-Sequential-NN-Classifier
A script that classifies an image as bright or dark, depending on the number of white pixels.

Part of an assignment for my intro to AI class, a neural network is trained to identify an image as either "bright" or "dark."
Assuming a 2x2 image size, if there are two or more white pixels, the image is bright. If there 1 or fewer white pixels, the image is dark. 
This script uitilizes the keras library for building a sequential model, which reads in the pixels of the image as an rgb value between 0 and 255. 
The inital training set is 7 images, while the testing set is 3. 
