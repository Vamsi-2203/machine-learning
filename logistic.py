import numpy as np
import pandas as pd
from matplotlib.pyplot import plot
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('datasets_185987_416528_diabetes2.csv')
x = data.iloc[:, 5:7].values
y = data.iloc[:, -1].values
#print(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_predict = classifier.predict(x_test)


print(y_predict)

cm = confusion_matrix(y_test, y_predict)
print(cm)