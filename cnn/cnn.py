# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

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

# first convolutional layer
# Step 1a - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 1b - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# second convolutional layer; to improve accuracy
# Step 2a - Convolution
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

# Step 2b - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 2 - Flattening
classifier.add(Flatten())

# Step 3 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

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

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # num of images in traiing set
                         #epochs = 25, # num of passes through neural network
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 2000) # num of images in test set