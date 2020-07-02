import sklearn
from sklearn import datasets, svm, metrics
import numpy as np

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test ,y_train, y_test = sklearn.model_selection.train_test_split(x, y , test_size=0.1)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_predict)
print(acc)