#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# File / Package Import
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#---------------------------------------------------------------------------------------------#
# standard library imports
#---------------------------------------------------------------------------------------------#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------------------------------------------------------#
# sklearn & sksurv imports
#---------------------------------------------------------------------------------------------#

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sksurv.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#---------------------------------------------------------------------------------------------#
# keras imorts
#---------------------------------------------------------------------------------------------#

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Data Preperation
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#---------------------------------------------------------------------------------------------#
# Importing the dataset
#---------------------------------------------------------------------------------------------#

dataset = pd.read_csv('Churn_Modelling.csv')
df_X = dataset.iloc[:, 3:13]
series_y = dataset.iloc[:, 13]
                
#---------------------------------------------------------------------------------------------#
# Encoding categorical data w/ sklearn encoder
#---------------------------------------------------------------------------------------------#

#le_gen = LabelEncoder()
#le_geo = LabelEncoder()
#df_X['Gender'] = le_gen.fit_transform(df_X['Gender'])
#df_X['Geography'] = le_geo.fit_transform(df_X['Geography'])
#
#ohe = OneHotEncoder(categorical_features = [1, 2], sparse = False)
#array_ohe_df_X = ohe.fit_transform(df_X)

#---------------------------------------------------------------------------------------------#
# Enconding categorical data w/ sksurv encoder
#---------------------------------------------------------------------------------------------#

df_X['Gender'] = df_X['Gender'].astype('category')
df_X['Geography'] = df_X['Geography'].astype('category')

ohe = OneHotEncoder(allow_drop = False)
df_ohe_X = ohe.fit_transform(df_X)

#---------------------------------------------------------------------------------------------#
# Splitting the dataset into the Training set and Test set
#---------------------------------------------------------------------------------------------#

X_train, X_test, y_train, y_test = train_test_split(
        df_ohe_X, series_y, test_size = 0.2, random_state = 0)

#---------------------------------------------------------------------------------------------#
# Feature Scaling
#---------------------------------------------------------------------------------------------#

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Modeling
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#---------------------------------------------------------------------------------------------#
# Initializing the ANN
#---------------------------------------------------------------------------------------------#

# create classifier
classifier = Sequential()

# add input layer and first hidden layer
classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu',
            input_dim = 11))

# add second hidden layer
classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu'))

# add output layer
classifier.add(
        Dense(
            units = 1, 
            kernel_initializer = 'uniform',
            activation = 'sigmoid'))

# compile ann
classifier.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])

#---------------------------------------------------------------------------------------------#
# Train ANN
#---------------------------------------------------------------------------------------------#

# Fitting classifier to the Training set
classifier.fit(
        x = X_train,
        y = y_train,
        batch_size = 10,
        epochs = 100)

#---------------------------------------------------------------------------------------------#
# Test ANN
#---------------------------------------------------------------------------------------------#

# this will give the probability of leaving
y_pred = classifier.predict(X_test)

#---------------------------------------------------------------------------------------------#
# Confusion matrix
#---------------------------------------------------------------------------------------------#

# convert probabilities into binary output; choose prob > 50% will leave bank
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Homework
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

'''
use the model to predict if the customer with the following characteristics
will leave the bank:
    Geography: France
    Credit Score: 600
    Gender: Male
    Age: 40
    Tenure: 3 years
    Balance: $60,000
    Number of Products = 2
    Customer has credit card: Yes
    Estimated Salary: $50,000

Ansswer:
    probability -> 0.0; customer will not leave
'''

# boolean arrays for characteristics
df_pred = pd.DataFrame(
        data = {'CreditScore':[600], 'Geography=Germany':[False], 'Geography=Spain':[False], 
                'Gender=Male':[True], 'Age':[40], 'Tenure':[3], 'Balance':[60000.],
                'NumOfProducts':[2], 'HasCrCard':[True], 'IsActiveMember':[True],
                'EstimatedSalary':[50000.]})

# scale pred
array_hw_pred = sc.transform(df_pred.values)

# predict
hw_pred = classifier.predict(array_hw_pred)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Model Evaluation
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#---------------------------------------------------------------------------------------------#
# classifier method for cross validation; copied from preivious text
#---------------------------------------------------------------------------------------------#

def build_classifier():
    classifier = Sequential()
    classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu',
            input_dim = 11))
    classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu'))
    classifier.add(
        Dense(
            units = 1, 
            kernel_initializer = 'uniform',
            activation = 'sigmoid'))
    classifier.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    return classifier

#---------------------------------------------------------------------------------------------#
# begin cross validation
#---------------------------------------------------------------------------------------------#

classifier_kc = KerasClassifier(
                    build_fn = build_classifier,
                    batch_size = 10,
                    epochs = 100)
accuracies = cross_val_score(
                estimator = classifier_kc,
                X = X_train,
                y = y_train,
                cv = 10,
                n_jobs = 1)
mean_accuracies = accuracies.mean()
var_accuracies = accuracies.std()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#
# Improving ANN
#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#---------------------------------------------------------------------------------------------#
# implementing dropout to prevent overfitting (large variance between train / test set)
#
# Initializing the ANN
#---------------------------------------------------------------------------------------------#

# create classifier
classifier = Sequential()

# add input layer and first hidden layer w/ droppout
classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu',
            input_dim = 11))
classifier.add(Dropout(rate = 0.1))

# add second hidden layer w/ droppout
classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

# add output layer
classifier.add(
        Dense(
            units = 1, 
            kernel_initializer = 'uniform',
            activation = 'sigmoid'))

# compile ann
classifier.compile(
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])

#---------------------------------------------------------------------------------------------#
# dropout; Train ANN
#---------------------------------------------------------------------------------------------#

# Fitting classifier to the Training set
classifier.fit(
        x = X_train,
        y = y_train,
        batch_size = 10,
        epochs = 100)

#---------------------------------------------------------------------------------------------#
# Parameter Tuning
#---------------------------------------------------------------------------------------------#

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu',
            input_dim = 11))
    classifier.add(
        Dense(
            units = 6, 
            kernel_initializer = 'uniform',
            activation = 'relu'))
    classifier.add(
        Dense(
            units = 1, 
            kernel_initializer = 'uniform',
            activation = 'sigmoid'))
    classifier.compile(
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = ['accuracy'])
    return classifier

#---------------------------------------------------------------------------------------------#
# Parameter Tuning; begin cross validation
#---------------------------------------------------------------------------------------------#

classifier_kc = KerasClassifier(
                    build_fn = build_classifier)
parameters = {
        'batch_size':[25, 32],
        'epochs':[100, 500],
        'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(
                estimator = classifier_kc,
                param_grid = parameters,
                scoring = 'accuracy',
                cv = 10)
grid_serach = grid_search.fit(X = X_train, y = y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

