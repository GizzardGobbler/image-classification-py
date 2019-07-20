# Image Classification With Python TensorFlow
An image classification program written in Python 3.6 using TensorFlow and Keras. Six image data sets are included. All images are from Google Image searches. I do not own any of the images. I am very young and inexperienced so feedback would be appreciated.

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

You must change the string 'PATH TO DATA SET' on lines 79 and 83 to the path were the project folder was saved.

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
