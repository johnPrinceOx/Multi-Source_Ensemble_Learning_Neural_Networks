"""
Main implementation of CNN-MSEL
J. Prince (c)
08/03/2018
john.prince@eng.ox.ac.uk


source ~/miniconda3/bin/activate deeplearn
"""
import numpy as N
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.optimizers import SGD
import scipy.io as spio
from keras.models import Model
import h5py
from keras.callbacks import ModelCheckpoint
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
import os
import usefulMethods as um
from keras.utils import plot_model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import nNetModelDefinitions as nnModels

# ###################
# Start the main loop
# ###################

for modelNum in range(1, 9):
    print("Current Model: " + str(modelNum))

    if modelNum == 2:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_learning_Model_" + modStr + "_trainTest.mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the training data
        modelData = h5py.File(modelDataString)  # Load Model Data File
        voiceMatrixAll = modelData["trainVoiceMat"]
        Y = modelData["trainVoiceY"]

        # Get the testing data
        voiceTest = modelData["testVoiceMat"]
        yTest = modelData["testVoiceY"]

        # Reshape the TESTING matrices
        x, y = voiceTest.shape
        B = N.reshape(voiceTest, [y * x])
        voiceTest = N.reshape(B, [y, x], [y, x])
        voiceTest = N.expand_dims(voiceTest, axis=2)

        # Reshape the TESTING response vector
        z, y = N.shape(yTest)
        yTest = N.reshape(yTest, [y, z], [y, z])

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 11):

            # Need to Balance the groups
            balancedIndices = um.balance_data(Y[0])
            voiceMatrix = voiceMatrixAll[:, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 2):
                print("Rep: " + str(rep) + " Fold: " + str(fold) + " Model: " + str(modelNum))
                trainIdx, valIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                voiceTrain = voiceMatrix[:, trainIdx]
                yTrain = Y[:, trainIdx]

                # Optional Augmentation of the training data
                #augData = um.data_augmentation_1D(voiceTrain)
                #voiceTrain = N.concatenate((voiceTrain, augData),axis=1)
                #yTrain =  N.concatenate((yTrain, yTrain),axis=1)

                # Validation Data
                voiceVal = voiceMatrix[:, valIdx]
                yVal = Y[:, valIdx]

                # Now build the model
                model_2 = nnModels.model_2_cnn(voiceMatrix)

                # Reshape the TRAINING matrices
                voiceX, voiceY = voiceTrain.shape
                B = N.reshape(voiceTrain, [voiceY * voiceX])
                voiceTrain = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceTrain = N.expand_dims(voiceTrain, axis=2)

                # Reshape the VALIDATION matrices
                voiceX, voiceY = voiceVal.shape
                B = N.reshape(voiceVal, [voiceY * voiceX])
                voiceVal = N.reshape(B, [voiceY, voiceX], [voiceY, voiceX])
                voiceVal = N.expand_dims(voiceVal, axis=2)

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the VALIDATION response vector
                z, y = N.shape(yVal)
                yVal = N.reshape(yVal, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [ModelCheckpoint('model_2_Weights_val.txt', monitor='val_loss', save_best_only=True, verbose=2)]

                history = model_2.fit(voiceTrain, yTrain, validation_data=(voiceVal, yVal),
                                      batch_size=150, # 150
                                      epochs=300,
                                      verbose=1,
                                      callbacks=callbacks)

                # Save the TRAINING results
                yPredTrain = model_2.predict(voiceTrain)
                trainResults = N.zeros((yTrain.shape[0], 2), dtype=N.float64)
                trainResults[:, 0] = yTrain.reshape(yTrain.shape[0])
                trainResults[:, 1] = yPredTrain.reshape(yTrain.shape[0])

                # Load best weights, predict, and save
                model_2.load_weights('model_2_Weights_val.txt')

                # Save the VALIDATION results
                yPredVal = model_2.predict(voiceVal)
                valResults = N.zeros((yVal.shape[0], 2), dtype=N.float64)
                valResults[:, 0] = yVal.reshape(yVal.shape[0])
                valResults[:, 1] = yPredVal.reshape(yVal.shape[0])

                # Save the TEST results
                yPredTest = model_2.predict(voiceTest)
                testResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                testResults[:, 0] = yTest.reshape(yTest.shape[0])
                testResults[:, 1] = yPredTest.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results_train = trainResults
                    model_results_val = valResults
                    model_results_test = testResults
                else:
                    model_results_train = N.concatenate([model_results_train, trainResults])
                    model_results_val = N.concatenate([model_results_val, valResults])
                    model_results_test = N.concatenate([model_results_test, testResults])

            spio.savemat('Model 2 CNN Results Final Train and val', mdict={'predictionsVal': model_results_val,
                                                                                'predictionsTrain': model_results_train,
                                                                                'predictionsTest': model_results_test})

    elif modelNum == 4:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_learning_Model_" + modStr + "_trainTest.mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File

        # Get the training/validation data
        walkingMatrixAll = modelData["trainWalkMat"]
        Y = modelData["trainWalkY"]

        # Get the testing data
        walkingTest = modelData["testWalkMat"]
        yTest = modelData["testWalkY"]
        print(walkingMatrixAll.shape)

        # Reshape the TESTING matrices
        walkingZ, walkingX, walkingY = walkingTest.shape
        B = N.reshape(walkingTest, [walkingY * walkingX * walkingZ])
        walkingTest = N.reshape(B, [walkingY, walkingX, walkingZ], [walkingY, walkingX, walkingZ])

        # Reshape the TESTING response vector
        z, y = N.shape(yTest)
        yTest = N.reshape(yTest, [y, z], [y, z])

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 11):

            # Balance the groups
            balancedIndices = um.balance_data(Y[0])
            walkingMatrix = walkingMatrixAll[:, :, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 11):
                print("Rep: " + str(rep) + " Fold: " + str(fold) + " Model: " + str(modelNum))
                trainIdx, valIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                walkingTrain = walkingMatrix[:, :, trainIdx]
                yTrain = Y[:, trainIdx]

                # Optional Augmentation of the training data
                #augData = um.data_augmentation_walk(walkingTrain)
                #walkingTrain = N.concatenate((walkingTrain, augData), axis=2)
                #yTrain = N.concatenate((yTrain, yTrain), axis=1)

                # Validation Data
                walkingVal = walkingMatrix[:, :, valIdx]
                yVal = Y[:, valIdx]

                # Implement the model
                model_4 = nnModels.model_4_cnn(walkingMatrix)

                # Reshape the Training matrices
                walkingZ, walkingX, walkingY = walkingTrain.shape
                B = N.reshape(walkingTrain, [walkingY * walkingX * walkingZ])
                walkingTrain = N.reshape(B, [walkingY, walkingX, walkingZ], [walkingY, walkingX, walkingZ])

                # Reshape the validation matrices
                walkingZ, walkingX, walkingY = walkingVal.shape
                B = N.reshape(walkingVal, [walkingY * walkingX * walkingZ])
                walkingVal = N.reshape(B, [walkingY, walkingX, walkingZ], [walkingY, walkingX, walkingZ])

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the Validation response vector
                z, y = N.shape(yVal)
                yVal = N.reshape(yVal, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [ModelCheckpoint('model_4_TESTWeights_test.txt', monitor='val_loss', save_best_only=True, verbose=1)]

                history = model_4.fit(walkingTrain, yTrain, validation_data=(walkingVal, yVal),
                                      batch_size=150,
                                      epochs=500,
                                      verbose=1,
                                      callbacks=callbacks)

                # Save the TRAINING results
                yPredTrain = model_4.predict(walkingTrain)
                trainResults = N.zeros((yTrain.shape[0], 2), dtype=N.float64)
                trainResults[:, 0] = yTrain.reshape(yTrain.shape[0])
                trainResults[:, 1] = yPredTrain.reshape(yTrain.shape[0])

                # Load best weights, predict, and save
                model_4.load_weights('model_4_TESTWeights_test.txt')

                # Save the VALIDATION results
                yPredVal = model_4.predict(walkingVal)
                valResults = N.zeros((yVal.shape[0], 2), dtype=N.float64)
                valResults[:, 0] = yVal.reshape(yVal.shape[0])
                valResults[:, 1] = yPredVal.reshape(yVal.shape[0])

                # Save the TEST results
                yPredTest = model_4.predict(walkingTest)
                testResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                testResults[:, 0] = yTest.reshape(yTest.shape[0])
                testResults[:, 1] = yPredTest.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results_train = trainResults
                    model_results_val = valResults
                    model_results_test = testResults
                else:
                    model_results_train = N.concatenate([model_results_train, trainResults])
                    model_results_val = N.concatenate([model_results_val, valResults])
                    model_results_test = N.concatenate([model_results_test, testResults])

            spio.savemat('Model 4 CNN Results Final Train and val', mdict={'predictionsVal': model_results_val,
                                                                        'predictionsTrain': model_results_train,
                                                                        'predictionsTest': model_results_test})

    elif modelNum == 8:
        modStr = str(modelNum)
        modelDataString = "MSL_Deep_learning_Model_" + modStr + "_trainTest.mat"  # Specify Model, Repetition, and Fold of Data File

        # Load the data
        modelData = h5py.File(modelDataString)  # Load Model Data File

        # Get the training/validation data
        tappingMatrixAll = modelData["trainTapMat"]
        Y = modelData["trainTapY"]

        # Get the testing data
        tappingTest = modelData["testTapMat"]
        yTest = modelData["testTapY"]

        # Reshape the TESTING matrices
        tappingZ, tappingX, tappingY = tappingTest.shape
        B = N.reshape(tappingTest, [tappingY * tappingX * tappingZ])
        tappingTest = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

        # Reshape the TESTING response vector
        z, y = N.shape(yTest)
        yTest = N.reshape(yTest, [y, z], [y, z])

        nSubs = Y.shape[1]
        print("There are " + str(nSubs) + " participants")
        nFolds = 10

        for rep in range(1, 11):

            #  Balance the groups
            balancedIndices = um.balance_data(Y[0])
            tappingMatrix = tappingMatrixAll[:, :, balancedIndices]
            Y = Y[:, balancedIndices]
            nBalSubs = len(balancedIndices)

            idx = um.cross_val_inds(nBalSubs, nFolds)

            for fold in range(1, 11):
                print("Rep: " + str(rep) + " Fold: " + str(fold) + " Model: " + str(modelNum))
                trainIdx, valIdx = um.kfold_train_test_idx(idx, fold)

                # Training Data
                tappingTrain = tappingMatrix[:, :, trainIdx]
                yTrain = Y[:, trainIdx]

                # Optional Augmentation of the training data
                # augData = um.data_augmentation_tap(tappingTrain)
                # tappingTrain = N.concatenate((tappingTrain, augData), axis=2)
                # yTrain = N.concatenate((yTrain, yTrain), axis=1)

                # Validation Data
                tappingVal = tappingMatrix[:, :, valIdx]
                yVal = Y[:, valIdx]

                # Implement the model
                model_8 = nnModels.model_8_cnn(tappingMatrix)

                # Reshape the TRAINING matrices
                tappingZ, tappingX, tappingY = tappingTrain.shape
                B = N.reshape(tappingTrain, [tappingY * tappingX * tappingZ])
                tappingTrain = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                # Reshape the Validation matrices
                tappingZ, tappingX, tappingY = tappingVal.shape
                B = N.reshape(tappingVal, [tappingY * tappingX * tappingZ])
                tappingVal = N.reshape(B, [tappingY, tappingX, tappingZ], [tappingY, tappingX, tappingZ])

                # Reshape the TRAINING response vector
                z, y = N.shape(yTrain)
                yTrain = N.reshape(yTrain, [y, z], [y, z])

                # Reshape the Validation response vector
                z, y = N.shape(yVal)
                yVal = N.reshape(yVal, [y, z], [y, z])

                # Now can fit the model
                # Set up the callbacks for saving the model with the best performance
                callbacks = [ModelCheckpoint('model8_TEST_train_Weights.txt', monitor='val_loss', save_best_only=True, verbose=1)]

                history = model_8.fit(tappingTrain, yTrain, validation_data=(tappingVal, yVal),
                                      batch_size=100,
                                      epochs=500,
                                      verbose=1,
                                      callbacks=callbacks)

                # Save the TRAINING results
                yPredTrain = model_8.predict(tappingTrain)
                trainResults = N.zeros((yTrain.shape[0], 2), dtype=N.float64)
                trainResults[:, 0] = yTrain.reshape(yTrain.shape[0])
                trainResults[:, 1] = yPredTrain.reshape(yTrain.shape[0])

                # Load best weights, predict, and save
                model_8.load_weights('model8_TEST_train_Weights.txt')

                # Save the VALIDATION results
                yPredVal = model_8.predict(tappingVal)
                valResults = N.zeros((yVal.shape[0], 2), dtype=N.float64)
                valResults[:, 0] = yVal.reshape(yVal.shape[0])
                valResults[:, 1] = yPredVal.reshape(yVal.shape[0])

                # Save the TEST results
                yPredTest = model_8.predict(tappingTest)
                testResults = N.zeros((yTest.shape[0], 2), dtype=N.float64)
                testResults[:, 0] = yTest.reshape(yTest.shape[0])
                testResults[:, 1] = yPredTest.reshape(yTest.shape[0])

                if (rep == 1) & (fold == 1):
                    model_results_train = trainResults
                    model_results_val = valResults
                    model_results_test = testResults
                else:
                    model_results_train = N.concatenate([model_results_train, trainResults])
                    model_results_val = N.concatenate([model_results_val, valResults])
                    model_results_test = N.concatenate([model_results_test, testResults])

            spio.savemat('Model 8 CNN Results Final Train and val', mdict={'predictionsVal': model_results_val,
                                                                           'predictionsTrain': model_results_train,
                                                                           'predictionsTest': model_results_test})

