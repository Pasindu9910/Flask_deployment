import pandas as pd
import numpy as np
import pickle

names = ['Overs','Runs','Wickets','Economy','Average','Bowler_st_rate','4s','6s','Dots','Training Module']
df = pd.read_csv('Training2.csv',names=names)
df.head()

x = df.iloc[:, :-1].values
y = df.iloc[:, 9].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.60)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

with open('knn_model2.pkl', 'wb') as file:
    pickle.dump(classifier, file)

