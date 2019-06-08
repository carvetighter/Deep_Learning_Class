# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

#---------------------------------------------------------------------------------------------#
# ??
#---------------------------------------------------------------------------------------------#

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# File / Package Import
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#---------------------------------------------------------------------------------------------#
# imports to fix PIL issue
#---------------------------------------------------------------------------------------------#

#import sys
#from PIL import Image
#sys.modules['Image'] = Image

#---------------------------------------------------------------------------------------------#
# keras imports
#---------------------------------------------------------------------------------------------#

from keras.models import Sequential
from keras.layers import Conv2D # because images are 2D
from keras.layers import MaxPooling2D # because images are 2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Build the CNN
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

'''
to imporove accuracy is to deepen the neural network; two options:
1. add convolutional layer(s)
    - choose this option
    - on second layer, steps 2a & 2b, do not need "imput_shape" parameter
    - cnn knows pooled feature maps from previous convolutional layer
2. add fully connected layer(s)
3. add both convolutional layer(s) and fully connected layer(s)
4. additional option is to add a third convolutional layer with 64 input parameters:
    - classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    - this is the combination of the two previous convolutional layers

Preprocessing Imagers (Keras Documentation) to reduce overfitting
ImageDataGenerator -> image transformation
    - flip, split, zoom, random transformations  on images
'''

# Initialising the CNN
classifier = Sequential()

# Step 1a - Convolution
'''
for convolutional layer, Conv2D():
filters = 32; integer; number of feature filters to use; standard practice is to start with 32
kernel_size = (3, 3); tuple; (rows, columns) of the feature filter
input_shape = (64, 64, 3); tuple; because we are using images and an image pre-processor the shape of the image is 
    64 rows / pixels by 64 columns / pixels; since this is a color image there are three channels Red, Green and Blue;
    channels is 3; (rows, columns, channels)
data_format = 'channels_last'; string; ensuring use Tensorflow format for "input_shape"
activation = 'relu'; string; use recifier function because classifying images are non-linear
'''
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), data_format = 'channels_last', activation = 'relu'))

# Step 1b - Pooling
'''
pooling the feature maps by creating a window, pool_size, in this case we are taking the maximum count of the
feature filter; this reduces the number of nodes we will have after flattening
pool_size = (2, 2); tuple; (reduction_factor_rows, reduction_factor_columns); 2 will reduce the dimension by half
'''
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 2a - Convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

# Step 2b - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Pre-process Images
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

train_datagen = ImageDataGenerator(rescale = 1./255, # make pixel image betweel 0 and 1
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Train the CNN
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#



#import Image
#print(Image.__file__)

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # num of images in training set
                         # epochs = 25, # num of passes through neural network
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 2000) # num of images in test set

# will swave to the working directory
classifier.save('cat_dog_image_cnn.h5')

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Make prediction
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

import numpy
from keras.preprocessing import image
from keras.models import load_model

dict_rev_class = dict()
for key, value in training_set.class_indices.items():
    dict_rev_class[value] = key

# load cnn if not used or present in memory
classifier = load_model('cat_dog_image_cnn.h5')

# test dog image
test_image_00 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image_00 = image.img_to_array(test_image_00)
test_image_00 = numpy.expand_dims(test_image_00, axis = 0)

array_result_00 = classifier.predict(test_image_00)
print(dict_rev_class.get(array_result_00[0][0]), 'None')

# test cat image
test_image_01 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image_01 = image.img_to_array(test_image_01)
test_image_01 = numpy.expand_dims(test_image_01, axis = 0)

array_result_01 = classifier.predict(test_image_01)
print(dict_rev_class.get(array_result_00[0][0]), 'None')