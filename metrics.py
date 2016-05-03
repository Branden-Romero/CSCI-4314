import scipy as sp

def spearman_scoring(clf, X, y):
    y_pred = clf.predict(X).flatten()
    return sp.stats.spearmanr(y_pred, y.flatten())[0]

def spearman_scoring_nn(network, X, y):
    y_pred = clf.predict(X).flatten()
    return sp.stats.spearmanr(y_pred, y.flatten())[0]
