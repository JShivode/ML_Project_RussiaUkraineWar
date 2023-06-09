import csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns

size = 100

def get_data():
    with open('../data/data_effect_on_fighting.csv', newline='') as csvfile:
        #date,temp,personnel,aircraft,helicopter,tank,APC,field_artillery,MRL,drone
        # creating a dictionary to hold the data
        data = {
            'date' : [],
            'temp': [],
            'personnel': [],
            'aircraft': [],
            'helicopter': [],
            'tank': [],
            'APC': [],
            'field_artillery': [],
            'MRL': [],
            'drone': [],
            }

        # creating a dict reader object
        data_file:dict = csv.DictReader(csvfile)
        for row in list(data_file):   # iterating through each row of the data
            for key , val in row.items(): # iterating through each key-value pair of the row
                data[key].append(val) # appending the value to the corresponding list in the dictionary
        df = pd.DataFrame(data) # creating a pandas dataframe from the dictionary
        X = np.asarray(df[['personnel', 'aircraft', 'helicopter', 'tank', 'APC', 'field_artillery', 'MRL', 'drone']])  # creating a numpy array for the features
        Y = np.asarray(df['temp']) # creating a numpy array for the target variable
    return X , Y # and returns the predictor variables 'X' and the target variable 'Y'.

#This function performs logistic regression using the data obtained from the
# get_data() function, which returns the feature matrix X and target vector Y.
def algo_logistic_regression():
    global size  # set the size variable as global
    X, Y = get_data()  # get data using the get_data function

    # initialize train and test score sums
    # which will be used to accumulate the training and testing scores over
    # multiple iterations of train/test splits and logistic regression.
    # The number of iterations is specified by the 'size' global variable.
    sum_train=0
    sum_test=0
    for x in range(size):  # perform train/test split and logistic regression for size number of iterations
        # split the data into training and testing sets (using 'train_test_split' function from 'sklearn'.)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle= True)

        # scale the training data (using the 'StandardScaler' function from 'sklearn.preprocessing'.)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        # initialize and train a logistic regression algorithm
        algo =  LogisticRegression(solver='liblinear', random_state=0)
        algo.fit(X_train, y_train)

        # scale the testing data and predict the output
        X_test = scaler.transform(X_test)
        y_pred = algo.predict(X_test)

        # calculate and store the test and train scores
        score_test = algo.score(X_test, y_test)
        score_train = algo.score(X_train, y_train)
        sum_train = sum_train + float(score_test)
        sum_test = sum_test + float(score_train)

    # calculate the average test and train scores and plot the results
    return plot_result(y_pred,y_test,sum_train/size,sum_test/size)

def algo_pca():
    global size
    X, Y = get_data() # get the data using the helper function get_data()
    sum_train = 0
    sum_test = 0
    for x in range(size):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True) # split the data into training and testing sets using train_test_split
        scaler = StandardScaler() # standardize the data using StandardScaler
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        pca = PCA(n_components=2) # apply PCA with two components
        # The PCA() function is used to reduce the dimensionality of the dataset to two dimensions.

        X_train = pca.fit_transform(X_train)  # fit the PCA model to the training data
        # The fit_transform() method is used to transform the training data into the new reduced dimension space.

        X_test = pca.transform(X_test) # transform the testing data into the same reduced dimension space as the training data.
        algo = LogisticRegression(solver='liblinear', random_state=0) # create a LogisticRegression object
        algo.fit(X_train, y_train) # fit the logistic regression model to the transformed training data
        y_pred = algo.predict(X_test) # make predictions on the transformed testing data
        score_test = algo.score(X_test, y_test) # calculate the accuracy of the model on the testing data
        score_train = algo.score(X_train, y_train) # calculate the accuracy of the model on the training data
        sum_train = sum_train + float(score_test) # add the accuracy of the model on the testing data to the running sum of training accuracies
        sum_test = sum_test + float(score_train)  # add the accuracy of the model on the training data to the running sum of testing accuracies
    return plot_result(y_pred,y_test,sum_train/size,sum_test/size) # plot the results and return them
    # The plot_result() function is called to create a plot that shows the predicted values against the actual values
    # of the target variable and the accuracy score of the model.

def algo_svm():
    global size # global variable that specifies the number of times to run the algorithm,
    X, Y = get_data() # input and output data
    # to store the accuracy scores:
    sum_train = 0
    sum_test = 0
    for x in range(size): # iterates the training and testing process over 'size' number of iterations

        #splits the data into training and testing sets with a test size of 0.2 and shuffled for each iteration.
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

        scaler = StandardScaler() # initializes the standard scaler for normalization of data.
        X_train = scaler.fit_transform(X_train) # applies the standard scaler to the training data.
        algo = SVC(kernel='linear', random_state=0) # initializes the SVM algorithm with a linear kernel and sets a random state of 0.
        algo.fit(X_train, y_train) # trains the algorithm with the training data.
        X_test = scaler.transform(X_test) # applies the same scaler to the testing data.
        y_pred = algo.predict(X_test) # predicts the target values using the trained algorithm.
        score_test = algo.score(X_test, y_test) # calculates the testing score of the algorithm.
        score_train = algo.score(X_train, y_train) # calculates the training score of the algorithm.
        sum_train = sum_train + float(score_test) # adds the testing score to 'sum_train'.
        sum_test = sum_test + float(score_train) # adds the training score to 'sum_test'.
    # returns the average training and testing scores as well as a plot
    # of the predicted vs. actual values using the plot_result() function.
    return plot_result(y_pred,y_test,sum_train/size,sum_test/size)

def algo_KNeighbors():
    global size
    X, Y = get_data() # assigned the data and labels using the 'get_data()' function
    sum=0
    # for x in range(size):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True) # split the data and labels into training and testing datasets.
    # 20% of the dataset is used for testing.
    # 'shuffle=True' is used to shuffle the dataset before splitting.

    neighbors = np.arange(1, 8) # generate a range of integers from 1 to 7.

    # empty arrays 'train_accuracy' and 'test_accuracy'
    # are created to store the training and testing accuracy scores respectively.
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over K values in the 'neighbors' array.
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k) # 'KNeighborsClassifier()' is used to create a k-nearest neighbors classifier with 'n_neighbors=k'.
        knn.fit(X_train, y_train) # train the classifier on the training data and labels.

        # Compute training and test data accuracy
        # (calculate the accuracy score of the classifier on both the training and testing datasets.)
        train_accuracy[i] = knn.score(X_train, y_train) #The training accuracy score is stored in the 'train_accuracy' array at the index 'i'.
        test_accuracy[i] = knn.score(X_test, y_test) # The testing accuracy score is stored in the 'test_accuracy' array at the index 'i'.

    # The training and testing accuracy scores are printed to the console.
    print('Score Train' ,train_accuracy)
    print('Score Test', test_accuracy)
    # Generate plot
    plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()

def plot_result(y_test,y_pred,score_train,score_test):
    print('Score Train', score_train)
    print('Score Test', score_test)
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

if __name__ == '__main__':
    print('Choose an algorithm: \n1.logistic_regression \n2.pca \n3.svm \n4.KNeighbors')
    x = input()

    if x=='1':
        algo_logistic_regression()
    elif x=='2':
        algo_pca()
    elif x=='3':
        algo_svm()
    elif x=='4':
        algo_KNeighbors()
