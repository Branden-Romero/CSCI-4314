import numpy as np

class Data:
	
	def __init__(self,data,splitpercent):
		self.X = self.get_x(data)
		self.y = self.get_y(data)
		self.X_train, self.y_train, self.X_test, self.y_test = self.splitdata(splitpercent,data)

	def get_x(self,data):
		return data[:,:-1]

	def get_y(self,data):
		return data[:,-1]
	
	def splitdata(self,splitpercent,data):
		numObjects = self.X.shape[0]
		numTestObjects = int(numObjects*splitpercent)
		testInd = np.random.choice(numObjects,numTestObjects)
		Xtest = self.X[testInd]
		ytest = self.y[testInd]
		trainInd = np.delete(np.array(range(numObjects)), testInd)
		Xtrain = self.X[trainInd]
		ytrain = self.y[trainInd]
		return (Xtrain,ytrain,Xtest,ytest)
