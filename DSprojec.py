import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import seaborn as sns

# Loading the data
data = pd.read_csv('weatherdata2.csv')


# Converting 'date' and 'time' columns to datetime
data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y")
data['time'] = pd.to_datetime(data['time'], format="%H:%M")

# Setting 'date' as the index
data.set_index('date', inplace=True)

# Extracting hour from 'time' column
data['hour'] = data['time'].dt.hour

# Droping the original 'time' column
data = data.drop(['time'], axis=1)

# Ensuring all values in the dataframe are non-negative
data[data < 0] = 0

#  data summary and information
print(data.describe(include='all'))
print(data.info())


fig = px.line(data, x=data.index, y='relative_humidity_2m (%)', title='Date vs Relative Humidity')
fig.show()

fig2 = px.line(data, x=data.index, y='Temperature(c)', title='Date vs Temperature')
fig2.show()

# Perform exploratory data analysis (EDA)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


plt.figure(figsize=(12, 10))

# for Temperature
plt.subplot(3, 3, 1)
plt.hist(data['Temperature(c)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Temperature(c)')
plt.ylabel('Frequency')
plt.title('Temperature Histogram')

# for Relative Humidity
plt.subplot(3, 3, 2)
plt.hist(data['relative_humidity_2m (%)'], bins=20, color='salmon', edgecolor='black')
plt.xlabel('relative_humidity_2m (%)')
plt.ylabel('Frequency')
plt.title('Relative Humidity Histogram')

# for Dew Point
plt.subplot(3, 3, 3)
plt.hist(data['dew_point(2m)'], bins=20, color='lightgreen', edgecolor='black')
plt.xlabel('dew_point(2m)')
plt.ylabel('Frequency')
plt.title('Dew Point Histogram')

# for Precipitation
plt.subplot(3, 3, 4)
plt.hist(data['precipitation (mm)'], bins=20, color='orange', edgecolor='black')
plt.xlabel('precipitation (mm)')
plt.ylabel('Frequency')
plt.title('Precipitation Histogram')


# for Pressure MSL
plt.subplot(3, 3, 6)
plt.hist(data['pressure_msl (hPa)'], bins=20, color='lightcoral', edgecolor='black')
plt.xlabel('pressure_msl (hPa)')
plt.ylabel('Frequency')
plt.title('Pressure MSL Histogram')

# for Surface Pressure
plt.subplot(3, 3, 7)
plt.hist(data['surface_pressure (hPa)'], bins=20, color='lightgrey', edgecolor='black')
plt.xlabel('surface_pressure (hPa)')
plt.ylabel('Frequency')
plt.title('Surface Pressure Histogram')

# for Wind Speed 10m
plt.subplot(3, 3, 8)
plt.hist(data['wind_speed_10m (km/h)'], bins=20, color='lightpink', edgecolor='black')
plt.xlabel('wind_speed_10m (km/h)')
plt.ylabel('Frequency')
plt.title('Wind Speed 10m Histogram')

plt.tight_layout()
plt.show()



y = data['Temperature(c)']  # Using temperature as the target variable for regression

# Defining feature set
X = data.drop(['Temperature(c)', 'relative_humidity_2m (%)'], axis=1)  

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying mutual information for feature selection
selector = SelectKBest(mutual_info_regression, k=4)
X_new = selector.fit_transform(X, y)

# Getting the selected features
mask = selector.get_support()
new_features = X.columns[mask]
print("Selected features:", new_features)

# Initializing and training Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

#  predictions using the trained model
y_pred = linear_model.predict(X_test)

# Evaluating the Linear Regression model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Displaying actual vs predicted values
print("Some actual and predicted values: ")
print("-Temperature")
df_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_results.head(10))  # Print the first 10 rows for comparison


# Printing evaluation metrics
print('Linear Regression:')
#average squared difference between the actual temperature values and the predicted values by the regression model
print(f'  Mean Squared Error (MSE): {mse:.2f}')
#statistical measure that represents the proportion of variance in the temperature data that is explained by the regression model.
print(f'  R-squared (R2): {r2:.4f}')

from sklearn.model_selection import cross_val_score

# Cross-validation with 5 folds
cv_scores = cross_val_score(linear_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse_scores = -cv_scores  # Convert negative MSE scores to positive
#MSE scores obtained through cross-validation, specifically using 5-fold cross-validation
print("Cross-Validation MSE Scores:", cv_mse_scores)
print("Mean CV MSE:", cv_mse_scores.mean())

# Plotting actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linestyle='--')  # Diagonal line representing perfect predictions
plt.xlabel('Actual Temperature (c)')
plt.ylabel('Predicted Temperature (c)')
plt.title('Actual vs Predicted Temperature')
plt.show()
