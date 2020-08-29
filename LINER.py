import pandas as pd
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

datasets = pd.read_csv("datasets_229906_491820_Fish.csv")
print(datasets.head())
x = datasets['Height'].values.reshape(-1, 1)
y = datasets['Width'].values.reshape(-1, 1)

#plot.plot(x, y, 'o')
#plot.show()

x_train, y_train = x[0: 160], y[0: 160]
x_test, y_test = x[160: ], y[160: ]

#print(x_train, y_train)

model = LinearRegression()
model.fit(x_train, y_train)

regression_line = model.predict(x)


r_sq = model.score(x, y)
print('cofficent:', r_sq)

print('intercept:', model.intercept_)
print('slope:', model.coef_)

plot.plot(x, regression_line)
plot.plot(x_train, y_train, 'o')
plot.plot(x_test, y_test, 'o')
plot.show()

print(regression_line)
