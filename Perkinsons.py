
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('E:/aiml/perkinsons.csv')
x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]].values
y = df.iloc[:, 17].values

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)


scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


'''logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
logreg.score(x_test, y_test)*100'''

#Accuracy = 87.75510204081633

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)*100

#Accuracy = 97.95918367346938

dt = DecisionTreeClassifier(random_state=1)
dt.fit(x_train, y_train)
dt.score(x_test, y_test)*100

#Accuracy =  83.6734693877551

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
rf.score(x_test, y_test)*100

#Accuracy = 95.91836734693877

res = knn.predict(x_train[[0]])
print(res)


print(y_train[[0]])

r = np.shape(x_train[[0]])

lsp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
print(np.shape(lsp))
rs = knn.predict([lsp])

np.shape(x_train)  #in 194 146 are in train and remaining are in test
np.shape(x_test)

res = knn.predict(x_test[[2]])

for i in range(146):
    r = knn.predict(x_train[[i]])
    print(r, y_train[i])

