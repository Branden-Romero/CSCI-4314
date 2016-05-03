import scipy as sp
import numpy as np

def spearman_scoring(clf, X, y):
	y_pred = clf.predict(X).flatten()
	return sp.stats.spearmanr(y_pred, y.flatten())[0]

def spearman_scoring_nn(network, testData):
	y_pred = []
	y_act = []
	for x,y in testData:
		y_pred.append(network.feedforward(x)[0][0])
		y_act.append(y[0][0])
	return sp.stats.spearmanr(y_pred, y_act)[0]
