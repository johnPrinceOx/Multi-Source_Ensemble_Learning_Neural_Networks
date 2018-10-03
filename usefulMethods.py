import numpy as N
import os
import matplotlib.pyplot as plt
import random
from sklearn.cross_validation import KFold, cross_val_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def balance_data(Y):
    """ 
        Method to balance binary datasets
        :ivar Y is the entire population response vector
        :returns balancedIndices: The indices of balanced datasets relative to original response vector
    """
    hcArray = N.nonzero(Y == 0)
    pdArray = N.nonzero(Y == 1)

    pdIdx = pdArray[0]
    hcIdx = hcArray[0]

    if len(hcIdx) > len(pdIdx):
        hcIdx = N.random.choice(hcIdx, len(pdIdx), replace=False)

    elif len(hcIdx) < len(pdIdx):
        pdIdx = N.random.choice(pdIdx, len(hcIdx), replace=False)

    balancedIndices = sorted(N.concatenate([pdIdx, hcIdx]))
    print('There are ' + str(len(pdIdx)) + ' PD participants')
    print('There are ' + str(len(hcIdx)) + ' HC participants')

    return balancedIndices


def cross_val_inds(nParticipants, nFold):
    """ 
    Method to generate cross validation indices
    nParticipants is the number of observations in the dataset
    nFold is the number of folds to be created
    """
    # First generate indices for the exact number division
    nReps = nParticipants//nFold
    firstIdx = N.random.permutation(N.repeat(range(10), nReps))
    firstIdx = N.asarray(firstIdx)

    # Then for the non exact divisions
    nExtra = nParticipants % nFold
    secondIdx = N.random.choice(nFold, nExtra, replace=False)
    secondIdx = N.asarray(secondIdx)

    # concatenate
    allIdx = N.concatenate((firstIdx, secondIdx), axis=0)
    allIdx = allIdx + 1
    allIdx = N.asarray(allIdx)
    return allIdx

def kfold_train_test_idx(allIdx, thisFold):
    """ 
    Method to generate the train and test indices using cross validation indices
    allIdx is a vector assigning each participant to a certain fold
    thisFold is the fold which we are testing (~=fold is the training)
    """
    trainIdx = [i for i, x in enumerate(allIdx) if x != thisFold]
    testIdx = [i for i, x in enumerate(allIdx) if x == thisFold]

    trainIdx = N.asarray(trainIdx)
    testIdx = N.asarray(testIdx)

    return trainIdx, testIdx


def data_augmentation_1D(trainingData):
    """
    Method to perform augmentation on every fifth and 7th training data (1Dimensional e.g. Voice)
    :param trainingData: 
    :return: augTraining 
    """
    nSamps = trainingData.shape[1]
    augTraining = N.empty([trainingData.shape[0], trainingData.shape[1]])
    print(augTraining.shape)

    for x in range(0,nSamps):

        if x % 5 == 0:
            thisData = trainingData[:, x]
            newData = N.multiply(-1.0, thisData)
            newData = -1*newData
            thisAug = [-1*p for p in newData]
            augTraining[:, x] = thisAug

        elif x % 7 != 0:
            thisData = trainingData[:, x]
            thisAug = thisData[::-1]
            augTraining[:, x] = thisAug

        else:
            thisData = trainingData[:, x]
            newData = N.multiply(-1.0, thisData)
            newData = -1 * newData
            thisAug = [-1 * p for p in newData]
            thisAug = thisAug[::-1]
            augTraining[:, x] = thisAug

    return augTraining


def data_augmentation_walk(trainingData):
    """
    Method to perform augmentation on every fifth and 7th training walking data instance
    :param trainingData: 
    :return: augTraining 
    """
    nSamps = trainingData.shape[2]
    augTraining = N.empty([trainingData.shape[0], trainingData.shape[1], trainingData.shape[2]])
    print(augTraining.shape)

    for x in range(0, nSamps):

        if x % 5 == 0:
            s1 = trainingData[0, :, x]
            s2 = trainingData[1, :, x]
            s3 = trainingData[2, :, x]
            s4 = trainingData[3, :, x]
            s5 = trainingData[4, :, x]
            s6 = trainingData[5, :, x]
            s7 = trainingData[6, :, x]
            s8 = trainingData[7, :, x]

            nS1 = N.multiply(-1.0, s1)
            nS2 = N.multiply(-1.0, s2)
            nS3 = N.multiply(-1.0, s3)
            nS4 = N.multiply(-1.0, s4)
            nS5 = N.multiply(-1.0, s5)
            nS6 = N.multiply(-1.0, s6)
            nS7 = N.multiply(-1.0, s7)
            nS8 = N.multiply(-1.0, s8)

            nS1 = -1*nS1
            thisAugS1 = [-1 * p for p in nS1]
            nS2 = -1 * nS2
            thisAugS2 = [-1 * p for p in nS2]
            nS3 = -1 * nS3
            thisAugS3 = [-1 * p for p in nS3]
            nS4 = -1 * nS4
            thisAugS4 = [-1 * p for p in nS4]
            nS5 = -1 * nS5
            thisAugS5 = [-1 * p for p in nS5]
            nS6 = -1 * nS6
            thisAugS6 = [-1 * p for p in nS6]
            nS7 = -1 * nS7
            thisAugS7 = [-1 * p for p in nS7]
            nS8 = -1 * nS8
            thisAugS8 = [-1 * p for p in nS8]

            augTraining[0, :, x] = thisAugS1
            augTraining[1, :, x] = thisAugS2
            augTraining[2, :, x] = thisAugS3
            augTraining[3, :, x] = thisAugS4
            augTraining[4, :, x] = thisAugS5
            augTraining[5, :, x] = thisAugS6
            augTraining[6, :, x] = thisAugS7
            augTraining[7, :, x] = thisAugS8

        elif x % 7 != 0:

            s1 = trainingData[0, :, x]
            s2 = trainingData[1, :, x]
            s3 = trainingData[2, :, x]
            s4 = trainingData[3, :, x]
            s5 = trainingData[4, :, x]
            s6 = trainingData[5, :, x]
            s7 = trainingData[6, :, x]
            s8 = trainingData[7, :, x]

            thisAugS1 = s1[::-1]
            thisAugS2 = s2[::-1]
            thisAugS3 = s3[::-1]
            thisAugS4 = s4[::-1]
            thisAugS5 = s5[::-1]
            thisAugS6 = s6[::-1]
            thisAugS7 = s7[::-1]
            thisAugS8 = s8[::-1]

            augTraining[0, :, x] = thisAugS1
            augTraining[1, :, x] = thisAugS2
            augTraining[2, :, x] = thisAugS3
            augTraining[3, :, x] = thisAugS4
            augTraining[4, :, x] = thisAugS5
            augTraining[5, :, x] = thisAugS6
            augTraining[6, :, x] = thisAugS7
            augTraining[7, :, x] = thisAugS8

        else:
            s1 = trainingData[0, :, x]
            s2 = trainingData[1, :, x]
            s3 = trainingData[2, :, x]
            s4 = trainingData[3, :, x]
            s5 = trainingData[4, :, x]
            s6 = trainingData[5, :, x]
            s7 = trainingData[6, :, x]
            s8 = trainingData[7, :, x]

            nS1 = N.multiply(-1.0, s1)
            nS2 = N.multiply(-1.0, s2)
            nS3 = N.multiply(-1.0, s3)
            nS4 = N.multiply(-1.0, s4)
            nS5 = N.multiply(-1.0, s5)
            nS6 = N.multiply(-1.0, s6)
            nS7 = N.multiply(-1.0, s7)
            nS8 = N.multiply(-1.0, s8)

            nS1 = -1 * nS1
            thisAugS1 = [-1 * p for p in nS1]
            nS2 = -1 * nS2
            thisAugS2 = [-1 * p for p in nS2]
            nS3 = -1 * nS3
            thisAugS3 = [-1 * p for p in nS3]
            nS4 = -1 * nS4
            thisAugS4 = [-1 * p for p in nS4]
            nS5 = -1 * nS5
            thisAugS5 = [-1 * p for p in nS5]
            nS6 = -1 * nS6
            thisAugS6 = [-1 * p for p in nS6]
            nS7 = -1 * nS7
            thisAugS7 = [-1 * p for p in nS7]
            nS8 = -1 * nS8
            thisAugS8 = [-1 * p for p in nS8]

            thisAugS1 = thisAugS1[::-1]
            thisAugS2 = thisAugS2[::-1]
            thisAugS3 = thisAugS3[::-1]
            thisAugS4 = thisAugS4[::-1]
            thisAugS5 = thisAugS5[::-1]
            thisAugS6 = thisAugS6[::-1]
            thisAugS7 = thisAugS7[::-1]
            thisAugS8 = thisAugS8[::-1]

            augTraining[0, :, x] = thisAugS1
            augTraining[1, :, x] = thisAugS2
            augTraining[2, :, x] = thisAugS3
            augTraining[3, :, x] = thisAugS4
            augTraining[4, :, x] = thisAugS5
            augTraining[5, :, x] = thisAugS6
            augTraining[6, :, x] = thisAugS7
            augTraining[7, :, x] = thisAugS8

    return augTraining


def data_augmentation_tap(trainingData):
    """
    Method to perform augmentation on every fifth and 7th training data point
    :param trainingData: 
    :return: augTraining 
    """
    nSamps = trainingData.shape[2]
    augTraining = N.empty([trainingData.shape[0], trainingData.shape[1], trainingData.shape[2]])
    print(augTraining.shape)

    for x in range(0, nSamps):

        if x % 2 == 0:
            s1 = trainingData[0, :, x]
            s2 = trainingData[1, :, x]
            s3 = trainingData[2, :, x]
            s4 = trainingData[3, :, x]
            s5 = trainingData[4, :, x]

            nS1 = N.multiply(-1.0, s1)
            nS2 = N.multiply(-1.0, s2)
            nS3 = N.multiply(-1.0, s3)
            nS4 = N.multiply(-1.0, s4)
            nS5 = N.multiply(-1.0, s5)

            nS1 = -1*nS1
            thisAugS1 = [-1 * p for p in nS1]
            nS2 = -1 * nS2
            thisAugS2 = [-1 * p for p in nS2]
            nS3 = -1 * nS3
            thisAugS3 = [-1 * p for p in nS3]
            nS4 = -1 * nS4
            thisAugS4 = [-1 * p for p in nS4]
            nS5 = -1 * nS5
            thisAugS5 = [-1 * p for p in nS5]

            augTraining[0, :, x] = thisAugS1
            augTraining[1, :, x] = thisAugS2
            augTraining[2, :, x] = thisAugS3
            augTraining[3, :, x] = thisAugS4
            augTraining[4, :, x] = thisAugS5

        elif x % 7 != 0:

            s1 = trainingData[0, :, x]
            s2 = trainingData[1, :, x]
            s3 = trainingData[2, :, x]
            s4 = trainingData[3, :, x]
            s5 = trainingData[4, :, x]

            thisAugS1 = s1[::-1]
            thisAugS2 = s2[::-1]
            thisAugS3 = s3[::-1]
            thisAugS4 = s4[::-1]
            thisAugS5 = s5[::-1]

            augTraining[0, :, x] = thisAugS1
            augTraining[1, :, x] = thisAugS2
            augTraining[2, :, x] = thisAugS3
            augTraining[3, :, x] = thisAugS4
            augTraining[4, :, x] = thisAugS5

        else:
            s1 = trainingData[0, :, x]
            s2 = trainingData[1, :, x]
            s3 = trainingData[2, :, x]
            s4 = trainingData[3, :, x]
            s5 = trainingData[4, :, x]

            nS1 = N.multiply(-1.0, s1)
            nS2 = N.multiply(-1.0, s2)
            nS3 = N.multiply(-1.0, s3)
            nS4 = N.multiply(-1.0, s4)
            nS5 = N.multiply(-1.0, s5)

            nS1 = -1 * nS1
            thisAugS1 = [-1 * p for p in nS1]
            nS2 = -1 * nS2
            thisAugS2 = [-1 * p for p in nS2]
            nS3 = -1 * nS3
            thisAugS3 = [-1 * p for p in nS3]
            nS4 = -1 * nS4
            thisAugS4 = [-1 * p for p in nS4]
            nS5 = -1 * nS5
            thisAugS5 = [-1 * p for p in nS5]

            thisAugS1 = thisAugS1[::-1]
            thisAugS2 = thisAugS2[::-1]
            thisAugS3 = thisAugS3[::-1]
            thisAugS4 = thisAugS4[::-1]
            thisAugS5 = thisAugS5[::-1]

            augTraining[0, :, x] = thisAugS1
            augTraining[1, :, x] = thisAugS2
            augTraining[2, :, x] = thisAugS3
            augTraining[3, :, x] = thisAugS4
            augTraining[4, :, x] = thisAugS5

    return augTraining




