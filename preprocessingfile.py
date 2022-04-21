
#Software Defect Prediction using CNN

#importing numpy
import numpy as np
#importing pandas
import pandas as pd
#importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
#importing train_test_split
from sklearn.model_selection import train_test_split
#importing matplotlib
import matplotlib.pyplot as plt
#importing recall_score
from sklearn.metrics import recall_score
#importing the sns package
import seaborn as sns
#importing under_sampling
from imblearn import under_sampling 
#importing sys
import sys
#importing over_sampling
from imblearn import over_sampling
from sklearn.preprocessing import MinMaxScaler
#importing SMOTE
from imblearn.over_sampling import SMOTE

#a dataset normalisation and feature selection function should be added
#calling to my_sdp_preprocessor with the dataset collected
def my_sdp_preprocessor(datafilename_as_csv_inquotes):
    #reading the original data file
    actual_data = pd.read_csv(datafilename_as_csv_inquotes)
    #removing the null values from the data
    actual_data.dropna(axis='columns')
    actual_data.isnull().values.any() #Gives false ie:No null value in dataset
    #Fill NA/NaN values using the specified method.
    actual_data = actual_data.fillna(value=False)
    actual_data = MinMaxScaler(missing_values = np.nan, strategy = 'mean')
    #actual_data sum for isnull valus=es
    actual_data.isnull().sum()
    #droping the defects column form the dataset
    original_X = pd.DataFrame(actual_data.drop(['defects'],axis=1))
    #adding defects dataset column to original_Y
    array = actual_data.values
    original_Y = actual_data['defects']
    #loading the dataframe to original_Y
    array = MinMaxScaler(feature_range=(0, 1))
    original_Y = pd.DataFrame(original_Y)
    #performing drop
    actual_data.dropna(axis='columns')
    i = 1
    while i < 6:
        #print(i)
        i += 1
    #splitting the dataset
    x_train1, x_test, y_train1, y_test= train_test_split(original_X, original_Y, test_size = .1,
                                                              random_state=12)
    from sklearn.impute import SimpleImputer
    # To replace the missing value we create below object of SimpleImputer class
    actual_data = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    #We now resample, and from that we generate training and validation data.
    array = actual_data.values
    sm = SMOTE(random_state=12, sampling_strategy = 1.0)
    #fir_resample of the data
    x, y = sm.fit_resample(x_train1, y_train1)
    array = MinMaxScaler(feature_range=(0, 1))
    #adding the defect column to y_train2
    #fitting the data
    actual_data.fit(x[:, 1:3])
    y_train2 = pd.DataFrame(y, columns=['defects'])
    #loading the original_X.columns to the x_train2
    x_train2 = pd.DataFrame(x, columns=original_X.columns)
    k = 1
    while i < 6:
        #print(i)
        k += 1
    #splliting the dataset infoto x_train2 and y_train2
    x_train, x_val, y_train, y_val= train_test_split(x_train2, y_train2, test_size = .1,
                                                              random_state=12)
    
    #copying the x_train data to combined_training data
    combined_training_data = x_train.copy()
    #putting the value of y_train data to combined training data
    combined_training_data['defects'] = y_train
    
    
    
    #combined training data to correction
    corr = combined_training_data.corr()
    combined_training_data.loc[:, :] = np.tril(combined_training_data, k=-1)
    cor_pairs = combined_training_data.stack()
    #loading to heatmap
    sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
    #getting the core_pair of actual data
    cor_pairs = np.tril(actual_data)
    #returning the data
    return actual_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val 

