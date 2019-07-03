# Neural Network
# by Trestan Simon
# 5 July 2019

# TODO
# > add more training data
# > organize code
# > add more datasets (maybe?)

# ---------------------------------------------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------------------------------------------


# Error message 'Invalid input'
def input_invalid():
    sys.exit("Input invalid.")


# Asks user which data set to use.
data_question = int(input("Which data set will be used?\n(1) Fly v. Clean (2) Maggot v. Rice (3) Baby v. Dough\n"), 10)
if data_question == 1:
    dataset = "fly-clean"
    class_names = ['fly', 'clean']
    file0_name = 'fly.*'
    file1_name = 'clean.*'
elif data_question == 2:
    dataset = "mag-rice"
    class_names = ['maggot', 'rice']
    file0_name = 'mag.*'
    file1_name = 'rice.*'
elif data_question == 3:
    dataset = "baby-dough"
    class_names = ['baby', 'dough']
    file0_name = 'baby.*'
    file1_name = 'dough.*'
else:
    input_invalid()


# Use Pillow library to convert an input jpeg to a 8 bit grey scale image array for processing.
def jpeg_to_8_bit_greyscale(path, maxsize):
    img = Image.open(path).convert('L')  # 'L' for grayscale 'RGB' for RGB
    width, height = img.size
    if width != height:
        m_min_d = min(width, height)
        img = img.crop((0, 0, m_min_d, m_min_d))
    # Scale the image to the requested maxsize by Anti-alias sampling.
    img.thumbnail(maxsize, PIL.Image.ANTIALIAS)
    return np.asarray(img)


def load_image_dataset(path_dir, maxsize):
    images = []
    labels = []
    os.chdir(path_dir)
    for file in glob.glob("*.jpg"):
        img = jpeg_to_8_bit_greyscale(file, maxsize)
        if re.match(file0_name, file):
            images.append(img)
            labels.append(0)
        elif re.match(file1_name, file):
            images.append(img)
            labels.append(1)
    return np.asarray(images), np.asarray(labels)


maxsize = 100, 100

(train_images, train_labels) = load_image_dataset(
    '/home/idstudent/PycharmProjects/TrestanSimon/FinalProject/' + dataset, maxsize)

(test_images, test_labels) = load_image_dataset(
    '/home/idstudent/PycharmProjects/TrestanSimon/FinalProject/' + dataset + '/test_set', maxsize)

print(train_labels)


def display_images(images, labels):
    plt.figure(figsize=(10, 10))
    grid_size = min(25, len(images))
    for i in range(grid_size):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])


# Where the checkpoint is saved
checkpoint_path =\
    '/home/idstudent/PycharmProjects/TrestanSimon/FinalProject/' + dataset + '/checkpoints/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

latest = tf.train.latest_checkpoint(checkpoint_dir)

display_question = input("\nWould you like to view the training data? (y/n): ").lower()
if display_question == "y" or display_question == "yes":
    display_images(train_images, train_labels)
    plt.show()
if display_question != "n" and display_question != "no":
    print("Input invalid.\nSkipping display.")

train_images = train_images / 255.0
test_images = test_images / 255.0

print(train_images.shape)

# ---------------------------------------------------------------------------------------------------------------

# Establishing model
model = keras.Sequential([
    # keras.layers.Flatten(input_shape=(100, 100)),
    keras.layers.Dense(1024, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(2, activation=tf.nn.softmax)])

# ---------------------------------------------------------------------------------------------------------------

# Checkpoint configuration
use_cp = input("Would you like to load from a checkpoint? (y/n): ").lower()
if use_cp == "y" or use_cp == "yes":
    cp_path_question = input(
        "Would you like to use the default checkpoint destination to load a checkpoint? (y/n): ").lower()
    if cp_path_question == "y" or cp_path_question == "yes":
        cp_path = '/home/idstudent/PycharmProjects/TrestanSimon/FinalProject/' + dataset + '/checkpoints/'
    elif cp_path_question == "n" or cp_path_question == "no":
        cp_path = input("Input file directory to load the checkpoint from: ")
    else:
        input_invalid()
    print("Looking for checkpoint in " + cp_path)
    cp_checker = path.exists(cp_path + 'cp-0000.ckpt.index')
    if cp_checker:
        print("A checkpoint has been found.")
        model.load_weights(latest)
        print("Successfully loaded from checkpoint.")
    else:
        print("A checkpoint has not been found.")
        sys.exit("Input invalid.\nIs the path correct? Does the file exist? Are you an idiot?")
elif use_cp != "n" and use_cp != "no":
    input_invalid()

# Epoch configuration
epoch_number = input("Input amount of epochs as an integer: ")
epoch_number = int(epoch_number, 10)
step_number = input("Input amount of steps per epoch as an integer: ")
step_number = int(step_number, 10)

# ---------------------------------------------------------------------------------------------------------------

sgd = keras.optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.7, nesterov=True)

model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, steps_per_epoch=step_number, epochs=epoch_number)

# ---------------------------------------------------------------------------------------------------------------

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
print(predictions)

display_images(test_images, np.argmax(predictions, axis=1))
plt.show()

# ---------------------------------------------------------------------------------------------------------------

# Save Checker
save_question = input("Model trained successfully.\nWould you like to save the model? (y/n): ").lower()
if save_question == "y" or save_question == "yes":
    model.save_weights(checkpoint_path.format(epoch=0))
    print("Model has been saved.")
elif save_question == "n" or save_question == "no":
    print("Model will not be saved.")
else:
    input_invalid()
