import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Sample data
weather_data = {
    'MONTH': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'AVG_TEMP': [36.0, 37.8, 46.9, 52.6, 63.0, 73.8, 77.8, 75.9, 69.6, 61.1, 52.4, 39.2],
    'PRECIPITATION': [5.84, 1.68, 9.88, 3.12, 2.62, 2.95, 2.15, 4.36, 0.98, 0.01, 3.14, 4.44]
}
delay_data = {
    'MONTH': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'TOTAL_WEATHER_DELAY': [13586.0, 4533.0, 6628.0, 6504.0, 8615.0, 13445.0, 9458.0, 18668.0, 3191.0, 1531.0, 1758.0, 6590.0]
}

# Merge the two datasets
weather_df = pd.DataFrame(weather_data)
delay_df = pd.DataFrame(delay_data)
merged_df = pd.merge(weather_df, delay_df, on='MONTH')

# Feature matrix and target
X = merged_df[['AVG_TEMP', 'PRECIPITATION']]  # 2D input
y = merged_df['TOTAL_WEATHER_DELAY']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline that standardizes the data and applies KNN
pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3))

# Fit the model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"KNN with StandardScaler - MSE: {mse:.2f}, R²: {r2:.2f}")

# Compare predictions to actual values
for i in range(len(y_test)):
    print(f"Predicted: {y_pred[i]:.1f}, Actual: {y_test.iloc[i]}")

# Optional: 2D plot using only temperature (can't visualize multi-dimensional nicely)
plt.figure(figsize=(10, 6))
plt.scatter(X_test['AVG_TEMP'], y_test, color='blue', label='Actual')
plt.scatter(X_test['AVG_TEMP'], y_pred, color='red', label='Predicted')
plt.xlabel('Average Temperature (°F)')
plt.ylabel('Total Weather Delay (min)')
plt.title('KNN Regression with Temp + Precipitation (2D View)')
plt.legend()
plt.grid(True)
plt.show()


# # 3D Graph
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot actual test data
# ax.scatter(X_test['AVG_TEMP'], X_test['PRECIPITATION'], y_test, c='blue', label='Actual', s=60)

# # Plot predicted values
# ax.scatter(X_test['AVG_TEMP'], X_test['PRECIPITATION'], y_pred, c='red', label='Predicted', s=60, marker='^')

# # Axis labels
# ax.set_xlabel('Average Temperature (°F)')
# ax.set_ylabel('Precipitation (in)')
# ax.set_zlabel('Total Weather Delay (min)')
# ax.set_title('KNN Regression: Temperature + Precipitation → Weather Delay')

# ax.legend()
# plt.tight_layout()
# plt.show()





# # Train and Predict on the whole dataset
# pipeline.fit(X, y)
# y_pred = pipeline.predict(X)

# plt.scatter(X['AVG_TEMP'], y, color='blue', label='Actual')
# plt.scatter(X['AVG_TEMP'], y_pred, color='red', label='Predicted')
# plt.legend()
# plt.show()
