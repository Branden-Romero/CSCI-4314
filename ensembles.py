import benchmark_data
import preprocess
import sklearn
from sklearn import linear_model
from sklearn import svm
from sklearn.grid_search import GridSearchCV
import sklearn.ensemble as en
import numpy as np
fileName = "FC_plus_RES_withPredictions.csv"
rd = preprocess.read_file(fileName)
data = preprocess.parse_crispr_data(rd)
sets = benchmark_data.Data(data,.4)

def spearman_scoring(clf, X, y):
    y_pred = clf.predict(X).flatten()
    return sp.stats.spearmanr(y_pred, y.flatten())[0]


#SVM
print("SVM")
parameters = {'kernel': ('linear', 'rbf'), 'C': np.linspace(1, 10, 10), 'gamma': np.linspace(1e-3, 1., 10)}
svr = svm.SVR(kernel="linear")
#SVM=GridSearchCV(svr,parameters,n_jobs=3,verbose=1,cv=10,scoring=spearman_scoring)

svr.fit(sets.X_train,sets.y_train)
#preprocess.pickle_data(SVM,"L1_linear")

#Random Forest
print("RF")
RF = en.RandomForestRegressor()
RF.fit(sets.X_train,sets.y_train)
preprocess.pickle_data(RF,"RF")

#Gradient Boosted RT
print("gbt")
GB = en.GradientBoostingRegressor()
GB.fit(sets.X_train,sets.y_train)
#preprocess.pickle_data(GB,"L1_linear")


	
	
	
