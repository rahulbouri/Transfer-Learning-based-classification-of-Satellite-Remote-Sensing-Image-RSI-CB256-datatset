# Classifying-Satellite-Remote-Sensing-Image-RSI-CB256-datatset-using-Transfer-Learning


## About the Dataset
For our classification task, we used the Satellite Remote Sensing Image-RSI-CB256 dataset. This is a readily available dataset that has RGB images (a total of 5631 images, with 3 channels) of varying sizes, from both sensor data and Google map snapshots, with each image classified into one of 4 classes: 
1. Cloudy
2. Water
3. Desert
4. Green Area

## Pre-Processing

Since the dataset included only images already focused on the aspect that is the basis for classi�cation, and does not as such involve spatial data or any other special formats, packages like EarthPy were not required for dataset loading and preprocessing. Only standard tensor�ow image manipulation functions were used.
Since, as shown above, the images in the dataset are of di�erent sizes, the images were rescaled to a standard size of (224, 224, 3). Apart from this, no other preprocessing was required for the application.

## Model

Transfer learning was used for classification. Specifically, VGG19 was used as the base of the model, with its standard ImageNet weights, and 0 retraining of VGG19 layers. In addition to this, standard layers of flatten, dense (1024 units) and dropout (with rate 0.3) were added to help improve classification accuracy. The dense layer was implemented with ReLu as its activation function. Finally, a dense layer with the softmax function for activation was used to get the final class assignments. Only these layers were maintained as trainable.

## Results

The highest training accuracy reached in 5 epochs was 0.9969. This was greatly due to the pretrained weights extracted from VGG19. The following shows the values of training loss and accuracy over the 5 epochs.

| Epoch | Training Loss      | Training Accuracy | 
| ----- | ------------------ | ----------- |
| 1. | 2.7011      | 0.8848      |
| 2. | 0.0455   | 0.9905        |
| 3. | 0.0168   | 0.9964        |
| 4. | 0.0405   | 0.9911        |
| 5. | 0.0105      | 0.9969       |
