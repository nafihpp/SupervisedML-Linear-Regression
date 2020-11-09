import pandas as pd
import numpy as np
import matplotlib.pyplot as pltimport 
get_ipython().run_line_magic('matplotlib', 'inline')


dataset = pd.read_csv('C:\student_scores.csv')
dataset.shape

dataset.head()

dataset.describe()

# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df
