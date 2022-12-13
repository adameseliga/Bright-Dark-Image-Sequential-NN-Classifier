import random
import keras
from keras import layers
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from math import sqrt
#######################################################################################################################
# matplotlib variables
COLOR = 'black'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
default_color_map = matplotlib.colors.ListedColormap(["black", "white"])
special_color_map_i_must_use_because_matplotlib_is_stubborn = matplotlib.colors.ListedColormap(["white", "black"])
#######################################################################################################################
# Neural Network Model


# image metadata
IMG_HEIGHT = 2
IMG_WIDTH = 2


# 6 hidden layers that use the sigmoid activation function
# The hidden last layer is twice the number of neurons than each of the previous hidden layers
model = keras.Sequential([
    layers.Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT)),
    layers.Dense(256, activation='sigmoid'),
    layers.Dense(256, activation='sigmoid'),
    layers.Dense(256, activation='sigmoid'),
    layers.Dense(256, activation='sigmoid'),
    layers.Dense(256, activation='sigmoid'),
    layers.Dense(512, activation='sigmoid'),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
#######################################################################################################################
# Preprocessing the Data


# import training/testing data from img_data directory
directory = 'img_data/'
CATEGORIES = ['bright', 'dark']

# Temporary data variables (eventually deleted)
training_data = []
testing_data = []
image_data = []


def createTrainingData():
    for category in CATEGORIES:
        # Read in images from the given directory
        path = os.path.join(directory, category)
        class_category = CATEGORIES.index(category)
        for image in os.listdir(path):
            # Convert image to grayscale, then into numpy array.
            image_array = cv2.imread(os.path.join(path, image), cv2.IMREAD_GRAYSCALE)
            image_array = np.array(cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT)))
            training_data.append(np.array([image_array, class_category]))


createTrainingData()
for data in training_data:
    image_data.append(data)    # Store images all images in separate list for the user to select

random.shuffle(training_data)   # Randomize data

# Training/Testing images and labels
training_images = []
training_labels = []
testing_images = []
testing_labels = []
images = []
image_labels = []

# Select 5 random images to test with
for i in range(5):
    random_index = random.randint(0, 14 - i)

    test_image = training_data.pop(random_index)
    testing_data.append(test_image)

# Separates the features and labels
for feature, label in image_data:
    images.append(feature)
    image_labels.append(label)
for feature, label in training_data:
    training_images.append(feature)
    training_labels.append(label)
for feature, label in testing_data:
    testing_images.append(feature)
    testing_labels.append(label)
# Delete these variables, as they are no longer needed.
del training_data, testing_data, image_data
# Because images have been converted to grayscale, the rgb value of any image is only one attribute.
# The next lines take the rgb value of all images and converts them into a floating point between 0 and 1.
training_images = np.array(training_images)  # Convert list of images to numpy array.
training_images = training_images / 255.
testing_images = np.array(testing_images)
testing_images = testing_images / 255.
images = np.array(images)
images = images / 255.

# Convert list of labels to numpy array
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)
image_labels = np.array(image_labels)
#######################################################################################################################
# Driver methods


def predict(model, image, correct_label):
    class_names = ['bright', 'dark']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)


# Count the number of bright pixels and return as integer
def countBrightPixels(image):
    # Selects a pixel from one of four quadrants
    img_border = int(sqrt(image.size) - 1)
    q_1 = image[0, img_border]
    q_2 = image[0, 0]
    q_3 = image[img_border, 0]
    q_4 = image[img_border, img_border]

    brightness_count = 0

    if q_1 == 1.:
        brightness_count += 1
    if q_2 == 1.:
        brightness_count += 1
    if q_3 == 1.:
        brightness_count += 1
    if q_4 == 1.:
        brightness_count += 1

    return brightness_count


# Chooses color map with respect to the edge cases (i.e. image is solid color)
# Returns plt.imshow method
def colorMap(mono_chrome):
    def chooseColorMap(*args):
        # If image is a solid color, use the custom color map
        if mono_chrome(*args):
            return plt.imshow(*args, cmap=special_color_map_i_must_use_because_matplotlib_is_stubborn)
        return plt.imshow(*args, cmap=default_color_map)

    return chooseColorMap


# Check if image is one color
@colorMap
def isMonoChrome(image):
    bright_pixels = countBrightPixels(image)
    if bright_pixels == 4 or bright_pixels == 0:
        return True
    return False


# Displays the image with its label
def show_image(image, label, guess):
    plt.figure()
    isMonoChrome(image)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


# Prompt user to pick a number, which selects an image from the img_data folder
def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num < 15:
                return int(num)
        else:
            print("Try again...")


# Driver
def main():
    while True:
        # Fits model to data
        model.fit(training_images, training_labels, epochs=500)
        test_loss, test_accuracy = model.evaluate(testing_images, testing_labels, verbose=1)
        print("Test Accuracy: ", test_accuracy)
        print("Test Loss: ", test_loss)
        num = get_number()
        image = images[num]
        image_label = image_labels[num]
        predict(model, image, image_label)
        user_input = None
        while user_input != 'y' and user_input != 'n':
            user_input = input("Continue?(Y/N): ").lower()
        if user_input == 'n':
            break


if __name__ == '__main__':
    main()
