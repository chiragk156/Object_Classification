'''
    assuming we have vgg_output.
    input for the svm_train Nx4096 matrix.
'''

#import cv2
import random as rn
import numpy as np
from svm_classification import read_vgg_output_data
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

def get_pred_acc(w,b,x,y):
    N = x.shape[0]
    pred_y = np.matmul(x,w) + b
    y = y.reshape((y.shape[0],1))
    pred_y[pred_y > 0] = 1
    pred_y[pred_y <= 0] = -1
    err = (50/float(N)) * np.sum(np.abs(pred_y - y))
    return 100 - err

 

def learn_svm_model(x, y, learning_rate = 0.2,C = 1, n_iters = 200):
    '''
        using unconstrained optimization problem over w.
        w : (4096,1)
        x : (N, 4096)
        y : (N,1)
        here y should have 1 or -1 as its entries.
        x[i] implies ith row.
    '''


    '''
    initialize w.
    '''
    N = x.shape[0]

    w = np.array([-2*rn.random() + 1 for i in range(4096)])
    w = w.reshape((4096,1))
    b = 0
    lamda = float(2)/(N*C)
    for iter in range(n_iters):
        #print(iter)
        '''
            step1 : forward pass that is find f(x).
            step2 : compute gradient of loss funtion w.r.t w and b.
            step3 : update w, b.

            for step1, f_x is (N,1) matrix.
            after that compute 1-yi*f(xi)
            for every ith row in d_psi, check if the 1-number is greater than 0 or not if it is less than 0 equate it to 0.

            d_psi is (N, 4096) = x*y every ith row of x is multiplied by yi after this make those rows 0 which result in 1-yi*f_xi < 0
            .This is done using d_mult (N,1). d_mask is like a mask which says when to take 0s and when to take y*x
            After this take summation of all the columns.
            dw = lamda*W + d_psi
        '''
        f_x = np.matmul(x,w)+b
        d_psi = -1 * (y*x)
        d_mult = 1 - y*f_x
        d_mask = np.zeros((N,1))
        for i in range(d_mult.shape[0]):
            if(d_mult[i][0] < 0):
                d_mask[i][0] = 0
            else:
                d_mask[i][0] = 1
        
        d_psi = d_mask * d_psi
        d_psi = np.sum(d_psi,axis=0).reshape((1,4096))
        d_psi = C * d_psi.T

        assert d_psi.shape == (4096,1)

        dw = lamda*w + (float(1)/N)*(d_psi)
        db = (float(1)/N)*(np.sum(d_mask * y,axis = 0))

        w = w - learning_rate*dw
        b = b - learning_rate*db
        error = (lamda) * (0.5 * (np.sum(w * w)) + (float(1)/N) * np.sum(d_mult, axis=0))
        #print ('error is ' + str(error))

    return w,b





def train_svm_classifier(x, y):
    '''
        this function is like a preproccessing for learn_svm_model function.
        class1 implies positive, class2 implies negative.
        class1 and class2 are strings.
        
    '''
    # vgg_features_output = '../vgg_features/'

    # classes = [class1, class2]
    # x,y = read_vgg_output_data(vgg_features_output,classes)
    y = y.reshape((y.shape[0],1))
    y[y == 1] = -1
    y[y == 0] = 1
    '''
        got x and y in desired format.
    '''
    svm_model = learn_svm_model(x,y)
    return svm_model 

def train_svm_ovo(classes):
    '''
        output is kc2 w,b.
        it is in the form of dictionary where key is tuple class indices and value is w,b tuple.
    '''

    vgg_features_output = '../vgg_features/'

    svm_classifiers = {}
    
    for i in range(len(classes)):
        for j in range(i+1,len(classes)):
            x,y = read_vgg_output_data(vgg_features_output,[classes[i],classes[j]])
            w,b = train_svm_classifier(x, y)
            svm_classifiers.update({(i,j) : (w,b)})
            print('done for ' + str(i) + ' ' + str(j))


    return svm_classifiers

def predict_svm(svm_model, x):
    w,b = svm_model
    f_x = np.dot(w.T, x) + b
    if(f_x > 0):
        '''
            implies positive class
        '''
        return 1
    else:
        '''
            implies negative class
        '''
        return 0

def predict_output(svm_classifiers, x, classes):
    output = {}
    for i in range(len(classes)):
        output.update({i : 0})

    for k in svm_classifiers:
        class_pos = k[0]
        class_neg = k[1]
        svm_model = svm_classifiers[k]
        output_class = predict_svm(svm_model, x)
        if(output_class == 1):
            output[class_pos] += 1
        else:
            output[class_neg] += 1
    max_key = max(output, key=lambda k: output[k])
    return max_key

def get_accuracy(svm_classifiers, x, y):
    '''
        x is (N, 4096), y is (N,1)
    
    '''
    N = float(x.shape[0])
    err_count = 0
    for i in range(int(N)):
        x_inp = x[i].reshape((4096,1))
        y_inp = y[i]
        y_pred = predict_output(svm_classifiers, x_inp, classes)

        if(y_inp != y_pred):
            err_count+=1

    error_perc = 100*(err_count/N)
    return 100 - error_perc




    


    
    



if __name__ == '__main__':
    vgg_features_output = '../vgg_features/'
    # x,y = read_vgg_output_data(vgg_features_output,classes)
    
    # svclassifier = SVC(kernel='linear',probability=True)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    # y_train[y_train == 1] = -1
    # y_train[y_train == 0] = 1
    # y_test[y_test == 1] = -1
    # y_test[y_test == 0] = 1
    # svclassifier.fit(X_train,y_train)
    # y_pred = svclassifier.predict(X_test)
    # # print(confusion_matrix(Y_train,Y_pred))
    # # print(classification_report(Y_train,Y_pred))
    
    # w,b = train_svm_classifier(X_train,y_train)
    # print(get_pred_acc(w,b,X_test,y_test))
    # print(get_pred_acc(w,b,X_train,y_train))
    # print('Testing Accuracy: ',accuracy_score(y_pred,y_test))


    classes = ['aeroplane','bicycle','bird','boat','car','person','horse','dog','cat']
    # svm_classifiers = train_svm_ovo(classes)
    # ifile = open('svm_weights','wb')
    # pickle.dump(svm_classifiers,ifile)
    # ifile.close()
    ifile = open('svm_weights','rb')
    svm_classifiers = pickle.load(ifile)
    ifile.close()
    x,y = read_vgg_output_data(vgg_features_output, classes)

    print(get_accuracy(svm_classifiers,x,y))