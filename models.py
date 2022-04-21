import pandas as pd
import keras
from tensorflow.keras.module.module import Sequential
from keras.layers import Dense
import preprocessingfile as preprocess
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D
from sklearn.svm import SVC


def svm(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    
    clf = SVC(gamma='auto')
    i = 1
    while i < 6:
        #print(i)
        i += 1
    clf.fit(x_train, y_train.values.reshape(-1,))
    return clf


def cnn(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    
    #create model
    
    x_train_matrix = x_train.values
    x_val_matrix = x_val.values
    y_train_matrix = y_train.values
    y_val_matrix = y_val.values
    
    from tensorflow.keras.datasets import cifar10
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras import backend as K
    
    
    ytrainseries = y_train['defects']
    #y_train_onehot = pd.get_dummies(ytrainseries)
    yvalseries = y_val['defects']
    #y_val_onehot = pd.get_dummies(yvalseries)
    
    img_rows, img_cols = 1,len(original_X.columns)
    from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten 
    from tensorflow.keras.layers import Reshape, Conv2DTranspose
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train1 = x_train_matrix.reshape(x_train_matrix.shape[0], img_rows, img_cols, 1)
    x_val1 = x_val_matrix.reshape(x_val_matrix.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    x_train_matrix = x_train.reshape(x_train.shape[0],img_rows,img_cols)
    x_train_matrix = x_test.reshape(x_test.shape[0],img_rows,img_cols)
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size=1, activation='relu',input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=1, activation='relu'))
    model.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = len(original_X.columns)))
    # Adding the second hidden layer
    model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
    model.add(Conv2D(16, kernel_size=1, activation='relu'))
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    model.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
    y_pred = model.predict(x_val)
    model.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
    model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    y_pred = model.predict(x_val)
    
#   model.add(MaxPool2D(pool_size=(1,8)))
    #model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    y_pred = model.predict(x_val)
    y_pred = (y_pred > 0.5)
    y_pred = pd.DataFrame(y_pred, columns=['defects'])
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #train the model
    model.fit(x_train1, y_train_matrix, epochs=40)   
    y_pred = model.predict(x_val)
    y_pred = model.predict(x_val1)>0.5
    model.fit(x_train, y_train.values.reshape(-1,))
    y_pred_df = pd.DataFrame(y_pred)
    
    return model         



    
def random_forest(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
    from keras.datasets import mnist
    #because our models are simple
    from keras.models import Sequential
    clf.fit(x_train, y_train.values.reshape(-1,))
    return clf


    
def NN(original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val):   
    # Importing the Keras libraries and packages
    
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = len(original_X.columns)))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
    classifier.add(Conv2D(64, kernel_size=1, activation='relu'))
    classifier.add(Conv2D(32, kernel_size=1, activation='relu'))
    classifier.add(Conv2D(16, kernel_size=1, activation='relu'))
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
    
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
    from keras.models import Sequential
    classifier.fit(x_train, y_train.values.reshape(-1,))
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
    from keras.datasets import mnist
    #because our models are simple
    from keras.models import Sequential
    classifier.add(Flatten())
    classifier.add(Dense(8, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))
    #compile model using accuracy to measure model performance
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    y_pred = classifier.predict(x_val)
    y_pred = pd.DataFrame(y_pred, columns=['defects'])
    y_pred = pd.predict(y_pred)>0.5
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
    classifier.add(Flatten())
    classifier.add(Dense(8, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))
    #compile model using accuracy to measure model performance
    #Making the predictions and evaluating the model
    # Predicting the Test set results
    y_pred = classifier.predict(x_val)
    from keras.datasets import mnist
    #because our models are simple
    from keras.models import Sequential
    y_pred = (y_pred > 0.5)
    y_pred = pd.DataFrame(y_pred, columns=['defects'])
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    classifier.add(Dense(8, activation='relu'))
    classifier.add(Dense(1, activation='sigmoid'))
    y_pred = classifier.predict(x_val)
    y_pred = (y_pred > 0.5)
    from __future__ import print_function
    y_pred = pd.DataFrame(y_pred, columns=['defects'])
    cm = confusion_matrix(y_val, y_pred)
    y_pred = pd.predict(y_pred)>0.5
    y_pred_df = pd.DataFrame(y_pred)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_val, y_pred)
    
    return classifier
        





    
#NN_clf = NN()
#rf_clf = random_forest()
#svm_clf = svm()
#cnn_clf = cnn()



