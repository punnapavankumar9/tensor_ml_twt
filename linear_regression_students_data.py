import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import pickle

data =  pd.read_csv('student-mat.csv', sep=";")

data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
predict = 'G3'
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y ,test_size=0.1)
# acc_each = 0
# for _ in range(30):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y ,test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#
#     if acc > acc_each:
#         print(acc)
#         with open("student-mat-trained-data.csv", 'wb') as f:
#             pickle.dump(linear, f)
#         acc_each = acc
pickle_in = open("student-mat-trained-data.csv", 'rb')
linear = pickle.load(pickle_in)


print("co-efficients: ", linear.coef_)
print("intercepts : ", linear.intercept_)

predictions = linear.predict(x_test)

for i in range(len(predictions)):
    print(predictions[i], x_test[i], y_test[i])