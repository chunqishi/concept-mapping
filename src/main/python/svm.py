from sklearn import datasets
from sklearn import svm
import numpy as np


iris = datasets.load_iris()
print iris.data.shape
print iris.target.shape

# linear_svc = svm.SVC(kernel='linear')
# rbf_svc = svm.SVC(kernel='rbf')

clf = svm.LinearSVC()
clf.fit(iris.data, iris.target)

res = clf.predict([[ 5.0,  3.6,  1.3,  0.25]])
print res

clf.coef_   