# Human-action-recognition
Human action recognition using Machine leaning and deploying to Comet.

Human Action Recognition (HAR) is a process of identifying and categorizing human actions from videos or image sequences. It is a challenging task in computer vision, and it has many practical applications, such as video surveillance, human-computer interaction, sports analysis, and medical diagnosis.

## Some use cases
- Healthcare.
- Surveillance and Security.
- Sports and Fitness.
- Robotics.
- Entertainment.

## Tools
- Python
- Comet
- VGG16 model

## The VGG model?
The VGG (Visual Geometry Group) model is a deep convolutional neural network architecture for image recognition tasks. It was introduced in 2014 by a group of researchers from the University of Oxford. 


The VGG model is known for its simplicity and effectiveness in image classification tasks. The model architecture consists of a series of convolutional layers, followed by max-pooling layers and finally, fully connected layers.

## Libraries

```Python
import os
import glob
import random
import numpy as np
import pandas as pd

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import to_categorical

import matplotlib.image as img
import matplotlib.pyplot as plt
```

## Display images

Happy Coding :)
