import numpy as np
import random
import os
from scipy.special import expit as sig
import cv2
import matplotlib.pyplot as plt
from svm_classification import read_vgg_output_data
from sklearn.model_selection import train_test_split
import pickle

# Dataset folder
DATASET_DIRECTORY = '../vgg_features/'
# validation split ratio
TEST_SPLIT_RATIO = 0.2

# Minibatch Size
MINIBATCH_SIZE = 100
# Learning Rate
LEARNING_RATE = 10000
# No. of epochs
EPOCHS = 1000
# Dropout rate
DROPOUT_RATE = 0


def read_data(dataset_dir,split_ratio):
	# Reading data.txt
	file = open(os.path.join(dataset_dir,'data.txt'))
	line = file.readline()

	# Data
	X = []
	Y = []
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []

	while line:
		[img_path, deg] = line.split('\t')

		img_path = os.path.join(dataset_dir,img_path)
		# removing '\n' from deg
		deg = float(deg[:-1])
		img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
		
		if img is None:
			line = file.readline()
			continue

		# Normalization
		img = (img - img.min())/img.max()

		X.append(img.flatten().T)
		Y.append(deg)
		line = file.readline()

	file.close()
	# Splitting data
	n = len(Y)
	r = random.sample(range(0, n), n)
	# Training data
	for i in r[:round(split_ratio*n)]:
		X_train.append(X[i])
		Y_train.append(Y[i])
	# Testing data
	for i in r[round(split_ratio*n):]:
		X_test.append(X[i])
		Y_test.append(Y[i])

	X_train = np.column_stack(tuple(X_train))
	Y_train = np.array(Y_train)
	X_test = np.column_stack(tuple(X_test))
	Y_test = np.array(Y_test)

	return X_train, Y_train, X_test, Y_test


class Neural_Network:
	
	def __init__(self, architecture):
		self.architecture = architecture
		self.n_layers = len(architecture)-1
		
		# Initializing random weights between -0.01 to 0.01
		self.weights = []
		for i in range(self.n_layers):
			# +1 for bias e.g. 1025x512
			w = np.zeros((architecture[i]+1,architecture[i+1]))
			for j in range(architecture[i]):
				w[j,:] = np.linspace(-1, 1, num=architecture[i+1])
			# w[1:,:] = (np.random.rand(architecture[i],architecture[i+1]) - 0.5)/50
			self.weights.append(w.T)



	def train_network(self, X_train, Y_train, X_test, Y_test, mini_batch_size, epochs, learning_rate, dropout_rate=0):
		N = len(Y_train)

		# Data shuffling
		# r = random.sample(range(0, N), N)
		# X_train = X_train[:,r]
		# Y_train = Y_train[r]

		# For plotting graph
		self.X_Graph = []
		self.Y_train_Graph = []
		self.Y_test_Graph = []
		# Running epochs
		for i in range(epochs):
			# For each minibatch
			for j in range(0, N, mini_batch_size):
				N_mini = min(j+mini_batch_size, N) - j
				X_mini = X_train[:,j:j+mini_batch_size]
				# Adding bias term
				X_mini = np.row_stack((np.ones((1, N_mini)), X_mini))
				Y_mini = Y_train[j:j+mini_batch_size]
				
				# Forward Pass
				# Z will store input, and output of every layer
				Z = [X_mini]
				for k in range(self.n_layers):
					# weight for kth layer
					w = self.weights[k]

					if dropout_rate>0 and dropout_rate<1:
						# Bias term is not effected
						Z[k][1:,] *= np.tile(np.random.binomial(1, 1 - dropout_rate, (Z[k].shape[0] - 1, 1)), mini_batch_size)
						
					# Output of kth layer Z[k+1] = w*Z[k]
					zk = w.dot(Z[k])
					# For last layer there is no activation function
					# Activation function: Sigmoid
					zk = sig(zk)
					if k != self.n_layers-1:
						# # Activation function: Sigmoid
						# zk = sig(zk)
						# Adding bias term
						zk = np.row_stack((np.ones((1, N_mini)), zk))

					Z.append(zk)

				# Backward Pass
				delta_weights = []
				for k in range(self.n_layers):
					delta_weights.append(np.zeros((self.architecture[k]+1,self.architecture[k+1])).T)

				# For last layer as it don't has activation function
				delta_z = (Z[self.n_layers]-Y_mini.T)*(Z[self.n_layers]*(1-Z[self.n_layers]))
				# delta_w = (output - y)*Z(K-1)
				# delta_weights[self.n_layers-1] = delta_z.dot(Z[self.n_layers-1].T)/N_mini
				delta_weights[self.n_layers-1] = (Z[self.n_layers-1].dot(delta_z.T)).T

				for k in range(self.n_layers-2,-1,-1):
					delta_z = Z[k+1]*(1-Z[k+1])*self.weights[k+1].T.dot(delta_z)
					# Removing bias term
					delta_z = delta_z[1:,:]

					# delta_weights[k] = delta_z.dot(Z[k].T)/N_mini
					delta_weights[k] = delta_z.dot(Z[k].T)
					# print(delta_weights[k])

				for k in range(self.n_layers):
					print(np.max(delta_weights[k]))
					self.weights[k] = self.weights[k] - learning_rate*delta_weights[k]

			training_Accuracy = self.Accuracy(X_train,Y_train)
			testing_Accuracy = self.Accuracy(X_test, Y_test)
			print(i+1,"\tTraining "+str(training_Accuracy)+"\tTesting "+str(testing_Accuracy))
			self.X_Graph.append(i+1)
			self.Y_train_Graph.append(training_Accuracy)
			self.Y_test_Graph.append(testing_Accuracy)



	def Accuracy(self,X_test, Y_test):
		# No of instances
		N = Y_test.shape[0]
		# Adding bias 1s
		X_test = np.row_stack((np.ones((1,N)),X_test))
		# Z will store output of every layer
		Z = X_test
		for k in range(self.n_layers):
			# weight for kth layer
			w = self.weights[k]

			zk = w.dot(Z)
			# For last layer there is no activation function
			# Activation function: Sigmoid
			zk = sig(zk)
			if k != self.n_layers-1:
				# Adding bias term
				zk = np.row_stack((np.ones((1, N)), zk))

			Z = zk

		Z = Z.T
		count = 0
		for i in range(Z.shape[0]):
			if np.argmax(Y_test[i]) == np.argmax(Z[i]):
				count += 1
		
		return count*100/N





# X_train, Y_train, X_test, Y_test = read_data(DATASET_DIRECTORY,SPLIT_RATIO)
# Read VGG outputs
classes = ['aeroplane','bicycle','bird','boat','car','person','horse','dog','cat']
X,y = read_vgg_output_data(DATASET_DIRECTORY,classes)
for i in range(X.shape[1]):
	if np.max(X[:,i])>0:
		X[:,i] = (X[:,i] - np.min(X[:,i]))/np.max(X[:,i])

X_train, X_test, Y_train1, Y_test1 = train_test_split(X, y, test_size = TEST_SPLIT_RATIO)
Y_train = np.zeros(shape=(Y_train1.shape[0],len(classes)))
Y_test = np.zeros(shape=(Y_test1.shape[0],len(classes)))
for i in range(Y_train1.shape[0]):
	Y_train[i,int(Y_train1[i])] = 1
for i in range(Y_test1.shape[0]):
	Y_test[i,int(Y_test1[i])] = 1
# Architecture
ARCHITECTURE = [4096,1024,256,64,len(classes)]
ann = Neural_Network(ARCHITECTURE)
ann.train_network(X_train.T,Y_train,X_test.T,Y_test,MINIBATCH_SIZE,EPOCHS,LEARNING_RATE,DROPOUT_RATE)