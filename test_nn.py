import network
from sklearn import svm
import sklearn.ensemble as en
from sklearn import linear_model
import preprocess
import data
import metrics
import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', type=str, help="Crispr dataset", required=True)
	args = parser.parse_args()
	
	fileName = args.f
	rawData = preprocess.read_file(fileName)
	crisprData = preprocess.parse_crispr_data(rawData)
	sets = data.Data(crisprData,.333)
	
	svr = svm.SVR()
	svr.fit(sets.X_train,sets.y_train)

	RF = en.RandomForestRegressor()
	RF.fit(sets.X_train,sets.y_train)

	GB = en.GradientBoostingRegressor()
	GB.fit(sets.X_train,sets.y_train)


	neuralNet = network.Network([36,18,9,1])
	neuralNet.SGD(sets.train, 100, 16, .1)

	#Linear Regression
	linear = linear_model.LinearRegression()
	linear.fit(sets.X_train,sets.y_train)

	#L2 Regression
	L2 = linear_model.Ridge(
		fit_intercept=True,
		normalize=True,
		copy_X=True, max_iter=None, tol=0.001, solver='auto')
	L2.fit(sets.X_train,sets.y_train)

	#L1 Regression
	L1 = linear_model.SGDRegressor('huber', epsilon=0.7,
		l1_ratio=1.0,
		fit_intercept=True,
		n_iter=10,penalty='elasticnet', shuffle=True)
	L1.fit(sets.X_train,sets.y_train)	

	print("DNN: ")
	print(metrics.spearman_scoring_nn(neuralNet,sets.train))
	print("Linear: ")
	print(metrics.spearman_scoring(linear,sets.X_test,sets.y_test))
	print("L2: ")
	print(metrics.spearman_scoring(L1,sets.X_test,sets.y_test))
	print("L1: ")
	print(metrics.spearman_scoring(L2,sets.X_test,sets.y_test))
	print("RF: ")
	print(metrics.spearman_scoring(RF,sets.X_test,sets.y_test))
	print("GB: ")
	print(metrics.spearman_scoring(GB,sets.X_test,sets.y_test))
	print("SVR: ")
	print(metrics.spearman_scoring(svr,sets.X_test,sets.y_test))
		
if __name__ == "__main__":
	main()
