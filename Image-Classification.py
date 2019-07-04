# Image Classification
# by Trestan
# 5 July 2019

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import os.path
from os import path
import re
import sys

import PIL
from PIL import Image


# Error message 'Invalid input'
def input_invalid():
    sys.exit("Input invalid.")


# Asks user which data set they want to use.
# User must input a number that corresponds to a data set.
data_question = int(input("Which data set will be used?\n(0) Fly v. Clean (1) Maggot v. Rice (2) Baby v. Dough\n"), 10)
if data_question == 0:
    dataset = "fly-clean"
    class_names = ['fly', 'clean']
    file0_name = 'fly.*'
    file1_name = 'clean.*'
elif data_question == 1:
    dataset = "mag-rice"
    class_names = ['maggot', 'rice']
    file0_name = 'mag.*'
    file1_name = 'rice.*'
elif data_question == 2:
    dataset = "baby-dough"
    class_names = ['baby', 'dough']
    file0_name = 'baby.*'
    file1_name = 'dough.*'
else:
    input_invalid()


# Converts an image to grey scale image array for processing.
def img_config(path, maxsize):
    img = Image.open(path).convert('L')  # coverts image to gray scale
    width, height = img.size
    if width != height:
        mini = min(width, height)  # Gets minimum values of image width and height
        img = img.crop((0, 0, mini, mini))  # Crops image
    img.thumbnail(maxsize, PIL.Image.ANTIALIAS)  # Scales images to maxsize variable by maxsize variable
    return np.asarray(img)


# Loads the image data set with their corresponding labels.
def load_image_dataset(path_dir, maxsize):
    images = []
    labels = []
    os.chdir(path_dir)
    for file in glob.glob("*.jpg"):
        img = img_config(file, maxsize)
        if re.match(file0_name, file):
            images.append(img)  # Appends img to the images array
            labels.append(0)  # Appends 0 to the labels array
        elif re.match(file1_name, file):
            images.append(img)  # Appends img to the images array
            labels.append(1)  # Appends 1 to the labels array
    return np.asarray(images), np.asarray(labels)


maxsize = 100, 100  # Maximum size of training and testing data

# Sets location of training data
(train_images, train_labels) = load_image_dataset(
    '/home/idstudent/PycharmProjects/TrestanSimon/FinalProject/' + dataset, maxsize)

# Sets location of testing data
(test_images, test_labels) = load_image_dataset(
    '/home/idstudent/PycharmProjects/TrestanSimon/FinalProject/' + dataset + '/test_set', maxsize)

print(train_labels)  # Prints training labels as  0s and 1s


# Displays the training data
def display_images(images, labels):
    plt.figure(figsize=(10, 10))
    grid_size = min(50, len(images))  # Grid size config
    for i in range(grid_size):
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])


# Where the checkpoint is saved
checkpoint_path = '/home/idstudent/PycharmProjects/TrestanSimon/FinalProject/' + dataset + '/checkpoints/'
checkpoint_path_file = checkpoint_path + 'cp-{epoch:04d}.ckpt'
checkpoint_path_os = os.path.dirname(checkpoint_path_file)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

latest = tf.train.latest_checkpoint(checkpoint_path_os)


# Asks user if they would like to view the training data.
display_question = input("Would you like to view the training data? (y/n): ").lower()
if display_question == "y" or display_question == "yes":
    display_images(train_images, train_labels)
    plt.show()
# Checks that user input doesn't equal no or yes.
elif display_question != "n" and display_question != "no" and display_question != "y" and display_question != "yes":
    print("Input invalid.\nSkipping display.")

train_images = train_images / 255.0
test_images = test_images / 255.0

# Prints training image shapes.
print("Training images shape: " + str(train_images.shape))


# Establishing model
model = keras.Sequential([
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation=tf.nn.softmax)])


# Checkpoint configuration
use_cp = input("Would you like to load from a checkpoint? (y/n): ").lower()
if use_cp == "y" or use_cp == "yes":
    print("Looking for checkpoint in " + checkpoint_path)
    cp_checker = path.exists(checkpoint_path_os)
    if cp_checker:
        print("A checkpoint has been found.")
        model.load_weights(latest)  # Loads the model from the latest checkpoint
        print("Successfully loaded from checkpoint.")
    else:
        print("A checkpoint has not been found.")
        input_invalid()
elif use_cp != "n" and use_cp != "no" and use_cp != "y" and use_cp != "yes":
    input_invalid()


# Epoch configuration
epoch_number = input("Input amount of epochs as an integer: ")
epoch_number = int(epoch_number, 10)
step_number = input("Input amount of steps per epoch as an integer: ")
step_number = int(step_number, 10)


sgd = keras.optimizers.SGD(lr=0.01, momentum=0.7, decay=1e-5, nesterov=True)  # Optimizer

model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Configures model for training

model.fit(train_images, train_labels, steps_per_epoch=step_number, epochs=epoch_number)  # Trains the model

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)  # Prints accuracy

predictions = model.predict(test_images) # Classifies test images
print(predictions)  # Prints predictions in percentage form

display_images(test_images, np.argmax(predictions, axis=1))  # Plots test images with what they're predicted to be
plt.show()


# Save Checker
save_question = input("Model trained successfully.\nWould you like to save the model? (y/n): ").lower()
if save_question == "y" or save_question == "yes":
    model.save_weights(checkpoint_path.format(epoch=0))  # Saves model
    print("Model has been saved.")
elif save_question == "n" or save_question == "no":
    print("Model will not be saved.")
else:
    input_invalid()
