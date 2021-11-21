# program to sun SVM on two different datasets and report results
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

#declare global variables for data
trainData, trainLabel, testData, testLabel = None, None, None, None
slackPenaltyRegulator = 1000
learningRate = 0.000001

#read the data from the cancer dataset
def readCancerFiles():
    global trainData, trainLabel, testData, testLabel

    #read attributes and labels data from cancer train
    trainData = pd.read_csv('./dataset_files/cancer_X_train.csv')
    trainLabel = pd.read_csv('./dataset_files/cancer_y_train.csv')

    #read attributes and label data from cancer test
    testData = pd.read_csv('./dataset_files/cancer_X_test.csv')
    testLabel = pd.read_csv('./dataset_files/cancer_y_test.csv')

#read the data from the iris data set
def readIrisData():
    global trainData, trainLabel, testData, testLabel

    #read attributes and labels data from cancer train
    trainData = pd.read_csv('./dataset_files/iris_X_train.csv')
    trainLabel = pd.read_csv('./dataset_files/iris_y_train.csv')

    #read attributes and label data from cancer test
    testData = pd.read_csv('./dataset_files/iris_X_test.csv')
    testLabel = pd.read_csv('./dataset_files/iris_y_test.csv')




# remove features that have little impact on the outcome


# filter features so that only significant features are used in the model
def chooseFeatures(trainData, trainLabel, testData):

    # remove features that are correlated because they will have similar 
    # effect of the outcome
    threshold = 0.9        # the threshold for correlation
    corr = trainData.corr()
    
    #initialize array
    columnsToBeDroppedCorrelation = np.full(corr.shape[0], False, dtype=bool)
    
    # find the columns numbers that need to be dropped
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= threshold:
                columnsToBeDroppedCorrelation[j] = True
    
    #finding the columns to be dropped
    columnsDroppedCorrelation = trainData.columns[columnsToBeDroppedCorrelation]
    
    #the columns from trainData and also testData to maintain consistency
    trainData.drop(columnsDroppedCorrelation, axis=1, inplace=True)
    testData.drop(columnsDroppedCorrelation, axis=1, inplace=True)

    # remove features that does not have significant impact on the outcome
    threshold_2, regOLS = 0.05, None

    columnsToBeDroppedSignificance = np.array([])

    #find columns to be dropped using ordinary least squares algorithm
    trainDataLen = len(trainData.columns)
    for itr in range(trainDataLen):
        regOLS = sm.OLS(trainLabel, trainData).fit()
        maximumCol = regOLS.pvalues.idxmax()
        maximumVal = regOLS.pvalues.max()
        if maximumVal > threshold_2:
            #drop the columns from both train and test to maintain consistency
            trainData.drop(maximumCol, axis='columns', inplace=True)
            testData.drop(maximumCol, axis='columns', inplace=True)
            columnsToBeDroppedSignificance = np.append(columnsToBeDroppedSignificance, [maximumCol])
        else:
            break


    return

# calculates the cost of the SVM
def calculateCost(W, X, Y):
    # calculate hinge loss
    slackVariable = 1 - Y * (np.dot(X, W))

    # select the maximum of 0 and distance for each item
    for i in range(len(slackVariable)):
        slackVariable[i] = max(0, slackVariable[i])

    # calculate the hinge loss by multiplying the slackPenaltyRegulator 
    # with the sum of the distances
    hingeLoss = slackPenaltyRegulator * (np.sum(slackVariable))

    # calculate the total value of the cost function
    totalCost = 1 / 2 * np.dot(W, W) + hingeLoss
    return totalCost

# finds the gradient of the cost function
def findGradient(W, Xi, Yi):

    # convert Xi and Yi to numpy array
    if type(Yi) == np.float64:
        Xi = np.array([Xi])  
        Yi = np.array([Yi])
    
    # calculate the distance
    distance = 1 - (Yi * np.dot(Xi, W))
    # initialize to all zeros
    dsum = np.zeros(len(W)) #value of gradient

    # value of the gradient of the cost function
    for i in range(len(distance)):
        #if the distance element is smaller than 0, di will be the same as W 
        if max(0, distance[i]) == 0:
            di = W
        else:
            di = W - (slackPenaltyRegulator * Yi[i] * Xi[i])
        dsum += di

    return dsum

# trains the models
def trainModel(features, outputs):   
    weights = np.zeros(features.shape[1]) #initialize weights array
    powerNum = 0             # used to check the cost when the iteration is a power of 2 (improves performance)
    prevCost = float("inf") # start with infinity as the previous cost
    changeThreshold = 0.01   # used to stop the training if the cost is not changing by a very large amount
    maximumNumberOfIterations = 4000 # maximum number of times the weights will be adjusted
    
    # print("Training model...")
    # train the model using stochastic gradient descent
    for iteration in range(1, maximumNumberOfIterations):
        # shuffling the features and outputs so that update cycles are not
        # repeated, while maintaining the order between them
        X, Y = shuffle(features, outputs)
        for j in range(X.shape[0]):
            gradient = findGradient(weights, X[j], Y[j])
            weights = weights - (learningRate * gradient)

        # check for convergence during the 2^powerNum iteration and during the last iteration
        if iteration == 2 ** powerNum or iteration == maximumNumberOfIterations - 1:
            cost = calculateCost(weights, features, outputs)
            # print("Iteration is: {} and Cost is: {}".format(iteration, cost))
            # if the cost did not change significantly compared to previous cost, return the weghts
            if abs(prevCost - cost) < changeThreshold * prevCost:
                # print("Model trained.\nWeights are: {}\nOptimized cost is: {}\n".format(weights, cost))
                return weights
            prevCost = cost
            powerNum += 1
    
    # print("Model trained.\nWeights are: {}\nOptimized cost is: {}\n".format(weights, cost))
    return weights

# run the model for the cancer data
def runForCancerData():

    global trainData, trainLabel, testData, testLabel
    # initialize the required data into the global variables
    readCancerFiles()

    for i in range(len(trainLabel['is_benign'])):
        if trainLabel['is_benign'][i] != 1:
            trainLabel['is_benign'][i] = -1

    chooseFeatures(trainData, trainLabel, testData)

    # normalise the features
    temp = MinMaxScaler().fit_transform(trainData.values)
    trainData = pd.DataFrame(temp)
    temp = MinMaxScaler().fit_transform(testData.values)
    testData = pd.DataFrame(temp)

    # insert 1 in every row for intercept b
    trainData.insert(loc=len(trainData.columns), column='intercept', value=1)
    # trainLabel.insert(loc=len(trainLabel.columns), column='intercept', value=1)
    testData.insert(loc=len(testData.columns), column='intercept', value=1)
    
    # convert the train table to numpy array for training 
    temp = trainLabel.to_numpy()
    tempArr = []
    for i in temp:
        if i[0] == 1:
            tempArr.append(float(1))
        else:
            tempArr.append(float(-1))

    trainLabelArr = np.array(tempArr)
    
    # train the model
    W = trainModel(trainData.to_numpy(), trainLabelArr)

    #convert the test label to numpy array
    temp = testLabel.to_numpy()
    tempArr1 = []
    for i in temp:
        if i[0] == 1:
            tempArr1.append(float(1))
        else:
            tempArr1.append(float(-1))

    testLabelArr = np.array(tempArr1)

    # test the model
    testLabelPredicted = np.array([])
    for i in range(testData.shape[0]):
        yp = np.sign(np.dot(testData.to_numpy()[i], W))
        testLabelPredicted = np.append(testLabelPredicted, yp)
    
    print("RESULTS for breast cancer dataset:")
    print("accuracy on test dataset: {}".format(accuracy_score(testLabelArr, testLabelPredicted)))
    print("recall on test dataset: {}\n".format(recall_score(testLabelArr, testLabelPredicted)))
    return

# find the weights for the given data
def FindIrisWeights(trainData, trainLabel, testData, classNumber):

    tempArr = []
    # convert train label to positive and negative class
    for i in range(len(trainLabel['Species'])):
        if trainLabel['Species'][i] == classNumber:
            trainLabel['Species'][i] = float(1)
            tempArr.append(float(1))
        else:
            trainLabel['Species'][i] = float(-1)
            tempArr.append(float(-1))

    chooseFeatures(trainData, trainLabel, testData)
    # normalise the features
    temp = MinMaxScaler().fit_transform(trainData.values)
    trainData = pd.DataFrame(temp)
    temp = MinMaxScaler().fit_transform(testData.values)
    testData = pd.DataFrame(temp)

    # insert 1 in every row for intercept b
    trainData.insert(loc=len(trainData.columns), column='intercept', value=1)
    testData.insert(loc=len(testData.columns), column='intercept', value=1)
    
    trainLabelArr = np.array(tempArr)

    W = trainModel(trainData.to_numpy(), trainLabelArr)

    return W, trainData, trainLabel, testData

# run the model for the iris data    
def runForIrisData():
    
    global trainData, trainLabel, testData, testLabel

    # read the data from files
    readIrisData()

    trainDataCopy = trainData.copy()
    trainLabelCopy = trainLabel.copy()
    testDataCopy = testData.copy()
    # print("1: ", trainLabelCopy)
    W0, traind0, trainl0, testd0 = FindIrisWeights(trainDataCopy, trainLabelCopy, testDataCopy, 0)

    # print("2: " , trainLabelCopy)
    trainDataCopy = trainData.copy()
    trainLabelCopy = trainLabel.copy()
    testDataCopy = testData.copy()
    # print(trainLabelCopy)
    W1, traind1, trainl1, testd1 = FindIrisWeights(trainDataCopy, trainLabelCopy, testDataCopy, 1)

    # check for class 1
    temp = testLabel.to_numpy()
    tempArr1 = []
    for i in temp:
        if i[0] == 1:
            tempArr1.append(float(1))
        else:
            tempArr1.append(float(-1))

    y_test = np.array(tempArr1)

    trainDataCopy = trainData.copy()
    trainLabelCopy = trainLabel.copy()
    testDataCopy = testData.copy()

    W2, traind2, trainl2, testd2 = FindIrisWeights(trainDataCopy, trainLabelCopy, testDataCopy, 2)

    # test the model
    y_test_predicted = np.array([])
    for i in range(testData.shape[0]):
        confidenceW0 = np.dot(testd0.to_numpy()[i], W0)
        confidenceW1 = np.dot(testd1.to_numpy()[i], W1)
        confidenceW2 = np.dot(testd2.to_numpy()[i], W2)

        maximumConfidence = max(confidenceW0, confidenceW1, confidenceW2)
        
        if (maximumConfidence == confidenceW0):
            y_test_predicted = np.append(y_test_predicted, 0)
        elif (maximumConfidence == confidenceW1):
            y_test_predicted = np.append(y_test_predicted, 1)
        else:
            y_test_predicted = np.append(y_test_predicted, 2)    

    temp = testLabel.to_numpy()
    tempArr1 = []
    for i in temp:
        tempArr1.append(float(i))

    y_test = np.array(tempArr1)

    print("RESULTS on Iris dataset:")
    print("accuracy on test dataset: {}\n".format(accuracy_score(y_test, y_test_predicted)))

    return

def main():
    global trainData, trainLabel, testData, testLabel

    # run the model for breast cancer data
    print("Running the model for breast cancer dataset")
    runForCancerData()

    # run the model for iris data
    print("Running the model for iris dataset")
    runForIrisData()

main()

