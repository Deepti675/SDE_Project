#importing pandas
import pandas as pd
#importing ensembler
import performance_check as ensembler
#importing preprocess
import preprocessingfile as preprocess
#importing all packages of sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import *
#loading data
data = 'pc2.csv'
#loading the data
log=LogisticRegression(penalty='l2',C=.01)
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val = preprocess.my_sdp_preprocessor(data)
original_data, original_X = preprocess.my_sdp_preprocessor(data)
#loading thefara
from sklearn.linear_model import LogisticRegression
all_data1 = [original_data, original_X, original_Y,combined_training_data,x_train1,x_train2]
all_data2 = [x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val]
X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])
nn_clf, cnn_clf, svm_clf, rf_clf = ensembler.send_classifiers_to_LR_file()
cnn_clf, svm_clf = ensembler.send_classifiers_to_LR_file()
log_reg_clf, new_test_set_x_matrix = ensembler.send_results_to_logistic_regression()

prediction = log_reg_clf.predict(new_test_set_x_matrix)
print('Accuracy:',accuracy_score(y_test.values,prediction))



#change 'data' variable in this file as well as performance_check.py
