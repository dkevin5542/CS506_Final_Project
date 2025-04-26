import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def test_model_on_airline_data(csv_path):
    # Load CSV
    test_df = pd.read_csv(csv_path)

    # Aggregate total weather_delay by (year, month)
    monthly_weather_delay = test_df.groupby(["year", "month"])["weather_delay"].sum().reset_index()

    # Simulate features (replace with real data if available)
    np.random.seed(42)
    monthly_weather_delay["precipitation"] = np.random.uniform(1.5, 5.5, size=len(monthly_weather_delay))
    monthly_weather_delay["storm_count"] = np.random.randint(1, 10, size=len(monthly_weather_delay))

    # Load trained scaler and model using dynamic relative path
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    scaler = joblib.load(os.path.join(model_dir, "jfk_scaler.pkl"))
    model = joblib.load(os.path.join(model_dir, "jfk_weather_model.pkl"))

    # Prepare features and predict
    X_test = monthly_weather_delay[["precipitation", "storm_count"]]
    X_scaled = scaler.transform(X_test)
    monthly_weather_delay["predicted_weather_delay"] = model.predict(X_scaled)

    return monthly_weather_delay


if __name__ == "__main__":
    # Path to your test data
    csv_path = "../data/clean/JFK_Weather_Delay_Data.csv"  # Update this if needed

    # Run prediction
    df = test_model_on_airline_data(csv_path)

    # Calculate error metrics
    true = df["weather_delay"]
    pred = df["predicted_weather_delay"]
    rmse = math.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)

    # Print metrics
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df["month"], true, label="Actual Weather Delay", marker="o", color="blue")
    plt.plot(df["month"], pred, label="Predicted Weather Delay", marker="^", color="red")
    plt.title(f"Predicted vs Actual Weather Delays (2024)\nRMSE = {rmse:.2f}, MAE = {mae:.2f}")
    plt.xlabel("Month")
    plt.ylabel("Delay (minutes)")
    plt.xticks(ticks=df["month"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

