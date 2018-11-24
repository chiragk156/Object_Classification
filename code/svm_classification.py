import pickle
import numpy as np
import imp
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split




def read_vgg_output_data(path,class_labels):
    X = None
    Y = None
    for i in range(len(class_labels)):
        file = open(path + class_labels[i],'rb')
        X_class = pickle.load(file)
        file.close()
        Y_class = np.ones(X_class.shape[0])*i
        if i == 0:
            X = X_class
            Y = Y_class
        else:
            X = np.concatenate((X, X_class))
            Y = np.concatenate((Y, Y_class))
    
    return X, Y

# Reading Data
# X,y = read_features_data(path)

def trainSVM(X,y,test_split_ratio):
    svclassifier = SVC(kernel='linear',probability=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_split_ratio)

    svclassifier.fit(X_train,y_train)
    y_pred = svclassifier.predict(X_test)
    # print(confusion_matrix(Y_train,Y_pred))
    # print(classification_report(Y_train,Y_pred))
    print('Testing Accuracy: ',accuracy_score(y_pred,y_test))
    return svclassifier