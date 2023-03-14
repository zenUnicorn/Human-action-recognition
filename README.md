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

```python
def DisplayImage():
    num = random.randint(1,10000)
    images = "Image_{}.jpg".format(num)
    train = "train/"
    if os.path.exists(train+images):
        testImage = img.imread(train+images)
        plt.imshow(testImage)
        plt.title("{}".format(train_dataset.loc[train_dataset['filename'] == "{}".format(images), 'label'].item()))

    else:
        print("File Path not found")
```

## Model 

```python
vgg_model = Sequential()

pretrained_model= tf.keras.applications.VGG16(include_top=False,
                   input_shape=(160,160,3),
                   pooling='avg',classes=15,
                   weights='imagenet')

for layer in pretrained_model.layers:
        layer.trainable=False

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(512, activation='relu'))
vgg_model.add(Dense(15, activation='softmax'))
```

## Test model

```python
# Function to predict

def test_predict(test_image):
    result = vgg_model.predict(np.asarray([read_image(test_image)]))

    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
    print("probability: "+str(np.max(result)*100) + "%\nPredicted class : ", prediction)

    image = img.imread(test_image)
    plt.imshow(image)
    plt.title(prediction)
```

Happy Coding :)
