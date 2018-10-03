"""
Model Definitions for CNN-MSL via mulit-task learning
J. Prince (c)
08/03/2018
john.prince@eng.ox.ac.uk
"""

import numpy as N
import pandas
import keras
from keras import optimizers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.layers import Embedding
from keras.layers import Conv1D, LSTM, MaxPooling1D
from keras.optimizers import SGD
import scipy.io as spio
from keras.models import Model
import h5py
from keras.layers.merge import concatenate
import os
from keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

""" Specify the structures of each of the three raw CNNs"""


def model_2_cnn(voiceMatrix):
    """ Model for target domain [0 0 1 0]"""

    # ####################
    # Reshape the matrices
    # ####################

    # Voice data is only 1D
    voiceMatrix = N.expand_dims(voiceMatrix, axis=0)
    somethingZ, voiceX, voiceY = voiceMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Voice Channel
    inputsVoice = Input(shape=(voiceX, 1))
    flatVoice = deepVoiceNet(inputsVoice, voiceX)

    # Merge the Features
    merged = flatVoice

    # Put Features into DNN
    finalOutputs = allDnn(merged)

    # Create & Compile Model
    model = Model(inputs=[inputsVoice], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def model_4_cnn(walkingMatrix):
    """ Model for target domain [0 1 0 0]"""

    # ####################
    # Reshape the matrices
    # ####################

    # For 3D Gait Data
    gaitZ, gaitX, gaitY = walkingMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Gait Channel
    inputsWalking = Input(shape=(gaitX, gaitZ))
    flatGait = deepWalkNet(inputsWalking, gaitX, gaitZ)

    # Merge the Features
    merged = flatGait

    # Put Features into DNN
    finalOutputs = allDnn(merged)

    # Create & Compile Model
    model = Model(inputs=[inputsWalking], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def model_8_cnn(tappingMatrix):
    """ Model for target domain [1 0 0 0]"""

    # ####################
    # Reshape the matrices
    # ####################

    # For 3D Tapping Data
    tapZ, tapX, tapY = tappingMatrix.shape

    # ########################
    # Build the Model Channels
    # ########################

    # Tapping CNN
    inputsTapping = Input(shape=(tapX, tapZ))
    flatTap = deepTapNet(inputsTapping, tapX, tapZ)

    # Merge the Features
    merged = flatTap

    # Put Features into DNN
    finalOutputs = allDnn(merged)

    # Create & Compile Model
    model = Model(inputs=[inputsTapping], outputs=finalOutputs)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model


def deepVoiceNet(inputsVoice, voiceX):
    l2_lambda = 1e-5

    # BRANCH 1
    # Layer 1
    b1 = Conv1D(filters=64,
                kernel_size=184000,
                strides=2700,
                input_shape=(voiceX, 1),
                kernel_regularizer=regularizers.l2(l2_lambda),
                kernel_initializer='he_normal')(inputsVoice)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling1D(pool_size=8, strides=4)(b1)
    b1 = Dropout(0.5)(b1)

    # 3x Convs
    for l in range(1, 4):
        b1 = Conv1D(filters=128,
                    kernel_size=8,
                    strides=1,
                kernel_initializer='he_normal')(b1)
        b1 = BatchNormalization()(b1)
        b1 = Activation('relu')(b1)
        # b1 = Dropout(drop)(b1)
    b1 = MaxPooling1D(pool_size=4, strides=4)(b1)

    # BRANCH 2
    b2 = Conv1D(64, 184000,
                strides=10000,
                input_shape=(voiceX, 1),
                kernel_regularizer=regularizers.l2(l2_lambda),
                kernel_initializer='he_normal')(inputsVoice)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = MaxPooling1D(pool_size=4, strides=1)(b2)
    b2 = Dropout(0.5)(b2)

    # 3x Convs
    for l in range(1, 3):
        print(l)
        b2 = Conv1D(filters=128,
                    kernel_size=6,
                    strides=1,
                    kernel_initializer='he_normal')(b2)
        b2 = BatchNormalization()(b2)
        b2 = Activation('relu')(b2)
        # b2 = Dropout(drop)(b2)
    b2 = MaxPooling1D(pool_size=2, strides=2)(b2)

    # Layer 9
    merged = concatenate([b1, b2], axis=1)
    merged = Dropout(0.5)(merged)
    flatVoice = Flatten()(merged)
    return flatVoice
    

def deepWalkNet(inputsWalking, gaitX, gaitZ):
    """ This method implements the same CNN and returns the features"""
    l2_lambda = 1e-5

    # BRANCH 1
    # Layer 1
    print(str(gaitX))
    print(str(gaitZ))
    b1 = Conv1D(filters=64,
                kernel_size=50,
                strides=6,
                input_shape=(gaitX, gaitZ),
                kernel_regularizer=regularizers.l2(l2_lambda),
                kernel_initializer='he_normal')(inputsWalking)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling1D(pool_size=8, strides=4)(b1)
    b1 = Dropout(0.5)(b1)

    # 3x Convs
    for l in range(1, 5):
        b1 = Conv1D(filters=128,
                    kernel_size=8,
                    strides=1,
                kernel_initializer='he_normal')(b1)
        b1 = BatchNormalization()(b1)
        b1 = Activation('relu')(b1)
        b1 = Dropout(drop)(b1)
    b1 = MaxPooling1D(pool_size=4, strides=4)(b1)

    # BRANCH 2
    b2 = Conv1D(filters=64,
                kernel_size=400,
                strides=25,
                input_shape=(gaitX, gaitZ),
                kernel_regularizer=regularizers.l2(l2_lambda),
                kernel_initializer='he_normal')(inputsWalking)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = MaxPooling1D(pool_size=4, strides=2)(b2)
    b2 = Dropout(0.5)(b2)

    # 3x Convs
    for l in range(1, 5):
        b2 = Conv1D(filters=128,
                    kernel_size=6,
                    strides=1,
                kernel_initializer='he_normal')(b2)
        b2 = BatchNormalization()(b2)
        b2 = Activation('relu')(b2)
        b2 = Dropout(drop)(b2)
    b2 = MaxPooling1D(pool_size=2, strides=2)(b2)

    # Layer 9
    merged = concatenate([b1, b2], axis=1)
    merged = Dropout(0.5)(merged)
    flatGait = Flatten()(merged)
    return flatGait
    

def deepTapNet(inputsTapping, tapX, tapZ):
    """ This method implements the same CNN and returns the features"""
    l2_lambda = 1e-5

    # BRANCH 1
    # Layer 1
    b1 = Conv1D(filters=64,
                kernel_size=50,
                strides=3,
                input_shape=(tapX, tapZ),
                kernel_regularizer=regularizers.l2(l2_lambda),
                kernel_initializer='he_normal')(inputsTapping)
    b1 = BatchNormalization()(b1)
    b1 = Activation('relu')(b1)
    b1 = MaxPooling1D(pool_size=8, strides=2)(b1)
    b1 = Dropout(0.5)(b1)

    # 3x Convs
    filtNum = [128, 128, 256, 256]
    for l in range(1, 5):
        b1 = Conv1D(filters=filtNum[l-1],
                    kernel_size=8,
                    strides=1,
                kernel_initializer='he_normal')(b1)
        b1 = BatchNormalization()(b1)
        b1 = Activation('relu')(b1)
        b1 = Dropout(drop)(b1)
    b1 = MaxPooling1D(pool_size=4, strides=4)(b1)

    # BRANCH 2
    b2 = Conv1D(filters=64,
                kernel_size=400,
                strides=10,
                input_shape=(tapX, tapZ),
                kernel_regularizer=regularizers.l2(l2_lambda),
                kernel_initializer='he_normal')(inputsTapping)
    b2 = BatchNormalization()(b2)
    b2 = Activation('relu')(b2)
    b2 = MaxPooling1D(pool_size=4, strides=2)(b2)
    b2 = Dropout(0.5)(b2)

    # 3x Convs
    filtNum = [128, 128, 256, 256]
    for l in range(1, 5):
        b2 = Conv1D(filters=filtNum[l-1],
                    kernel_size=20,
                    strides=1,
                kernel_initializer='he_normal')(b2)
        b2 = BatchNormalization()(b2)
        b2 = Activation('relu')(b2)
        b2 = Dropout(drop)(b2)
    b2 = MaxPooling1D(pool_size=2, strides=2)(b2)

    # Layer 9
    merged = concatenate([b1, b2], axis=1)
    merged = Dropout(0.5)(merged)
    flatTap = Flatten()(merged)
    return flatTap
    
def allDnn(merged):
    l2_lambda = 1e-5

    x = Dense(200, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(merged)
    x = Dropout(0.5)(x)
    x = Dense(300, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = Dropout(0.5)(x)
    x = Dense(50, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda))(x) 
    x = Dropout(0.5)(x)
    finalOutputs = Dense(1, activation='sigmoid')(x)
    return finalOutputs
