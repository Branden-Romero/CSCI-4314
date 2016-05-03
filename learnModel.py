import benchmark_data
import preprocess
from sklearn import linear_model
import numpy as np
fileName = "FC_plus_RES_withPredictions.csv"
rd = preprocess.read_file(fileName)
data = preprocess.parse_crispr_data(rd)
sets = benchmark_data.Data(data,.4)


#Linear Regression
linear = linear_model.LinearRegression()
linear.fit(sets.X_train,sets.y_train)
preprocess.pickle_data(linear,"linear")

#L2 Regression
L2 = linear_model.Ridge(
	alpha=1e-3,
	fit_intercept=True, 
	normalize=True, 
	copy_X=True, max_iter=None, tol=0.001, solver='auto')
L2.fit(sets.X_train,sets.y_train)
preprocess.pickle_data(L2,"L2_linear")

#L1 Regression
L1 = linear_model.SGDRegressor('huber', epsilon=0.7, 
	alpha=1e-3,
	l1_ratio=1.0, 
	fit_intercept=True, 
	n_iter=10,penalty='elasticnet', shuffle=True, 
	normalize=True)
L1.fit(sets.X_train,sets.y_train)
preprocess.pickle_data(L1,"L1_linear")
