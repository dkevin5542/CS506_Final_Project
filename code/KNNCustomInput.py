import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

## Todo: 1. Change the model to Linear Regression or SVM's
#        2. Change the code so that it does train test split using our data and then in a different file
#           create our own features manually and see how many minutes of delay per month we are going to have
#        3. Change the data so that it imports the data instead of doing it manually
#        4. Maybe create a front end so that when we input like avg temp or avg percipitation per month our
#           model will return the average delay in JFK

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

# Merge the datasets
weather_df = pd.DataFrame(weather_data)
delay_df = pd.DataFrame(delay_data)
merged_df = pd.merge(weather_df, delay_df, on='MONTH')

# Train on all data
X = merged_df[['AVG_TEMP', 'PRECIPITATION']]
y = merged_df['TOTAL_WEATHER_DELAY']

# Standardize the features manually
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KNN model
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X_scaled, y)

# Custom test inputs
custom_inputs = pd.DataFrame({
    'AVG_TEMP': [40, 50, 60, 70, 80],
    'PRECIPITATION': [2.0, 3.0, 1.0, 4.0, 5.0]
})
custom_inputs_scaled = scaler.transform(custom_inputs)

# Predict delays
predictions = model.predict(custom_inputs_scaled)

# Print predictions
print("\nCustom Predictions:")
for i in range(len(custom_inputs)):
    t = custom_inputs.iloc[i]['AVG_TEMP']
    p = custom_inputs.iloc[i]['PRECIPITATION']
    d = predictions[i]
    print(f"Temp: {t}°F, Precip: {p}in → Predicted Delay: {d:.1f} min")

# 2D Plot (Temp vs Predicted Delay)
plt.figure(figsize=(10, 6))
plt.scatter(X['AVG_TEMP'], y, color='blue', label='Training Data')
plt.scatter(custom_inputs['AVG_TEMP'], predictions, color='red', label='Predicted Points', marker='^')
plt.xlabel('Average Temperature (°F)')
plt.ylabel('Total Weather Delay (min)')
plt.title('KNN Predicted Weather Delays Based on Temp + Precip')
plt.legend()
plt.grid(True)
plt.show()


# # 3D Plot
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot original data
# ax.scatter(X['AVG_TEMP'], X['PRECIPITATION'], y, color='blue', label='Training Data', s=50)

# # Plot custom predictions
# ax.scatter(custom_inputs['AVG_TEMP'], custom_inputs['PRECIPITATION'], predictions,
#            color='red', marker='^', s=70, label='Predicted Points')

# ax.set_xlabel('Average Temperature (°F)')
# ax.set_ylabel('Precipitation (in)')
# ax.set_zlabel('Predicted Delay (min)')
# ax.set_title('KNN Predictions for Custom Weather Inputs')
# ax.legend()
# plt.tight_layout()
# plt.show()
