# -*- coding: utf-8 -*-

# Imports


# Basics 
import os 
import matplotlib.pyplot as plt 
import seaborn as sns

# Data processing 
import numpy as np
import pandas as pd 

# ML functions 
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, Embedding
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 
from tensorflow.keras.utils import plot_model

"""# Dataset

Mount drive
"""

from google.colab import drive
drive.mount('/content/gdrive')

dataPath = '/content/gdrive/MyDrive/data/'

"""Short data summary"""

count = 1 
dSum = 0 
for l in os.listdir(dataPath):
    subName = os.path.join(dataPath, l) 
    print(str(count) + ". Class name: " + l + ", Number of images = " + str(len(os.listdir(subName)))) 
    count += 1 
    dSum += len(os.listdir(subName))

print("Total = " + str(dSum))

print("Printing random examples for each of the 4 classes in the dataset used \n")
fig = plt.figure(figsize = (10, 8))
count = 1 
sizes = []
for l in os.listdir(dataPath):
    subName = os.path.join(dataPath, l)
    # print(len(os.listdir(subName)), np.random.randint(0, len(os.listdir(subName)), size = 1))
    randInd = np.random.randint(0, len(os.listdir(subName)), size = 1)
    imageName = os.path.join(subName, os.listdir(subName)[randInd[0]]) 
    img = img_to_array(load_img(imageName))
    sizes.append(img.shape) 
    
    plt.subplot(4, 1,count)
    plt.imshow(load_img(imageName))
    plt.title(f"{l}")
    plt.axis('off')
    count += 1
plt.show()

count = 1 
for l in os.listdir(dataPath):
    subName = os.path.join(dataPath, l) 
    print(str(count) + ". Class name: " + l + ", Image size = " + str(sizes[count - 1])) 
    count += 1

"""Since the images have different sizes, (but each with 3 channels for RGB) preprocessing to resize all images is required

# Preprocessing
"""

targetSize = 224
batchSize = 64

datagen = ImageDataGenerator(rescale = 1 / 255, validation_split = 0.2)

trainGen = datagen.flow_from_directory(directory = dataPath, batch_size = batchSize, class_mode = 'categorical', target_size = (targetSize, targetSize), subset = 'training') 

testGen = datagen.flow_from_directory(directory = dataPath, batch_size = batchSize, class_mode = 'categorical', target_size = (targetSize, targetSize), subset = 'validation')

"""# Models

## Inception V3 Load
"""

!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

from tensorflow.keras.applications.inception_v3 import InceptionV3

def incLoad(): 
  localWeights = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
  inc = InceptionV3(input_shape = (targetSize, targetSize, 3), include_top = False, weights = None)
  inc.load_weights(localWeights) 
  for layers in inc.layers:
    layers.trainable = False 
  lastOutput = inc.get_layer('mixed7').output 

  return inc, lastOutput

"""## VGG19 Load"""

def vggLoad(): 
  vgg = tf.keras.applications.vgg19.VGG19(input_shape = (targetSize, targetSizeS, 3), include_top = False, weights = "imagenet") 
  for layers in vgg.layers:
    layers.trainable = False 
  lastOutput = vgg.layers[-1].output 

  return vgg, lastOutput

"""## Model Definition"""

dense = 512
dropoutRate = 0.3

def defineLayer(lastOutput): 
  layer = tf.keras.layers.Flatten()(lastOutput)
  layer = tf.keras.layers.Dense(dense, activation = 'relu')(layer)
  layer = tf.keras.layers.Dropout(dropoutRate)(layer)
  layer = tf.keras.layers.Dense(4, activation = 'softmax')(layer) 

  return layer

modelChoice = "vgg" # ""inc" or "vgg"

if modelChoice == "inc": 
  inc, lastOutput = incLoad() 
  layer = defineLayer(lastOutput)
  model = Model(inputs = inc.input, outputs = layer) 
else: 
  vgg, lastOutput = incLoad() 
  layer = defineLayer(lastOutput)
  model = Model(inputs = vgg.input, outputs = layer)

model.compile(optimizer = 'adam', loss = tf.keras.metrics.categorical_crossentropy, metrics = ['accuracy'])

plot_model(model, to_file='/content/gdrive/MyDrive/satClass' + modelChoice + '.png', show_shapes = True, show_layer_names = True)

"""# Training"""

history = model.fit(trainGen, 
                    epochs = 5,
                    verbose = 1)

"""# Evaluation

## Plotting training accuracy
"""

# Plot the training and validation accuracies for each epoch

acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r')
# plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training accuracy')
plt.legend(loc=0)
plt.figure() 

plt.show()

"""## Evaluation on test dataset"""

score = model.evaluate_generator(testGen, verbose = 1)
print('Loss on test dataset : ', np.round(score[0], 4))
print('Accuracy on test dataset : ', np.round(score[1], 4))

