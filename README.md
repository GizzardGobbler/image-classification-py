> [!WARNING]  
> I created this repository in July 2019 when I was 14 years old. I was relatively new to Python at the time, so the code is not written to my current standards. Nonetheless, I have made this repository public as a record of my past machine learning and Python experience.

# Image Classification With Python TensorFlow

An image classification program written in Python 3.6 using TensorFlow and Keras. Six image data sets are included. All images are from Google Image searches. I do not own any of the images.

## Imports
```
import tensorflow as tf
from tensorflow import keras

import numpy as np
import maptlotlib.pyplot as plt
import glob
import os
import os.path
from os import path
import re
import sys

import PIL
from PIL import Image
```
## Set Up

You must change the string 'PATH TO DATA SET' on lines 79 and 83 to the path inside of the project folder.

```
# Sets location of training data
(train_images, train_labels) = load_image_dataset(
    'PATH TO DATA SET' + dataset, maxsize)

# Sets location of testing data
(test_images, test_labels) = load_image_dataset(
    'PATH TO DATA SET' + dataset + '/test_set', maxsize)
```

## Problems
At times, especially using the fly and clean data sets, memory allocation errors occur.
There are a lot of problems. I'll expand this section in the future.
