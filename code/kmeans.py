import numpy as np
import imp
import random as rn
from svm_classification import read_vgg_output_data
from sklearn.cluster import KMeans
from main import vgg_features_output
import pickle

#kmeans clustring call
def kmeans(data,n_clusters):
	kmeans = KMeans(n_clusters,random_state=0).fit(data)
	return kmeans
		

#making confusion matrix
def confusion_matrix(kmeans,data,clusters,labels,labels_no=9):
	cm = np.zeros((clusters,labels_no)) #creating initial 0 value 2d confusion array 
	(a,b) = data.shape
	for i in range(a):
		cm[kmeans.labels_[i]][int(labels[i])]+=1
	return cm	



#create mapping from maximum occurence
def create_mapping(kmeans,data,clusters,labels,label_no=9):
	cm = confusion_matrix(kmeans,data,clusters,labels,label_no)
	m = {}
	for i in range(clusters):
		maximum=0
		for j in range(label_no):
			if (cm[i][j]>maximum):
				maximum = cm[i][j]
				m[i] = j
	return m			

#finding accuracy
def accuracy(data,labels,kmeans,mapping):
	(a,b) = data.shape
	correct = 0					#no of correct labels
	predicted_values = kmeans.predict(data)		#predicted values
	for i in range(a):
		if (mapping[predicted_values[i]] == labels[i][0]):
			correct+=1
	return 100*(correct/a)

if __name__=="__main__":
    classes = ['aeroplane','bicycle','bird','boat','car','person','horse','dog','cat']
    clusters = len(classes)

    x,y = read_vgg_output_data(vgg_features_output,classes)
    y = y.astype(int)
    y = y.reshape((y.shape[0],1))

    # kmeans = kmeans(x,clusters)
    # ifile = open('kmeans','wb')
    # pickle.dump(kmeans,ifile)
    # ifile.close()
    
    ifile = open('kmeans','rb')
    kmeans = pickle.load(ifile)
    ifile.close()

    print('kmeans stored')
    m = create_mapping(kmeans,x,clusters,y,len(classes))
    print(accuracy(x,y,kmeans,m))
				