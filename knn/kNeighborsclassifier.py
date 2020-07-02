import sklearn
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("car.data")
le = preprocessing.LabelEncoder()

buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
door = le.fit_transform(list(data['door']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

predict = "class"
X = list(zip(buying, maint, persons, lug_boot, door, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.02)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)
predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for i in range(len(predicted)):
    if names[predicted[i]] != names[y_test[i]]:
        print(names[predicted[i]], x_test[i], names[y_test[i]])

    n = model.kneighbors([x_test[i]], 9, True)
    print("N: ", n)