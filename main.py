import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# load data
data = pd.read_csv('data/sales.csv')

# convert date to month
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month

# input and output
X = data[['month']]
y = data['sales']

# train model
model = LinearRegression()
model.fit(X, y)

# predict future
future = [[7], [8], [9]]
predictions = model.predict(future)

print("Predictions:", predictions)

# plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Forecast")
plt.show()



df = pd.DataFrame({
    'month': [7, 8, 9],
    'predicted_sales': predictions
})

df.to_csv('output/predictions.csv', index=False)