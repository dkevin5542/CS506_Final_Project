# train_model.py (Random Forest version)

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math


def load_and_clean_data():
    """Load and preprocess updated JFK weather delay, precipitation, and storm event data."""
    delay_df = pd.read_csv("../data/clean/JFK_Weather_Delay_Data21.csv")
    precip_df = pd.read_csv("../data/raw/raw data/JFKWeatherData/monthly_total_precipitation21.csv", skiprows=2)
    storm_df = pd.read_csv(
        "../data/raw/raw data/JFKWeatherData/storm_data_search_results21.csv",
        on_bad_lines='skip',
        engine='python'
    )

    delay_df["airport"] = delay_df["airport"].str.strip()
    jfk_df = delay_df[delay_df["airport"] == "JFK"].reset_index(drop=True)
    monthly_delay = jfk_df.groupby(["year", "month"])["weather_delay"].sum().reset_index()

    precip_df.columns = precip_df.columns.str.strip()
    precip_df = precip_df[pd.to_numeric(precip_df["Year"], errors="coerce").notna()]
    precip_df["Year"] = precip_df["Year"].astype(int)
    precip_df = precip_df.drop(columns=["Annual"], errors="ignore")
    precip_long = precip_df.melt(id_vars=["Year"], var_name="month", value_name="precipitation")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_mapping = {month: idx + 1 for idx, month in enumerate(months)}
    precip_long["month"] = precip_long["month"].map(month_mapping)
    precip_long = precip_long.rename(columns={"Year": "year"})


    storm_df["BEGIN_DATE"] = pd.to_datetime(storm_df["BEGIN_DATE"], errors="coerce")
    storm_df["year"] = storm_df["BEGIN_DATE"].dt.year
    storm_df["month"] = storm_df["BEGIN_DATE"].dt.month

    # Clean the EVENT_TYPE column
    storm_df["EVENT_TYPE"] = storm_df["EVENT_TYPE"].str.strip().str.title()

    # Then filter
    relevant_types = ["High Wind", "Thunderstorm Wind", "Flash Flood", "Winter Weather", "Strong Wind", "Heavy Snow", "Hail"]
    storm_df = storm_df[storm_df["EVENT_TYPE"].isin(relevant_types)]

    # Now count
    storm_counts = storm_df.groupby(["year", "month"]).size().reset_index(name="storm_count")

    merged = monthly_delay.merge(precip_long, on=["year", "month"], how="inner")
    merged = merged.merge(storm_counts, on=["year", "month"], how="left")
    merged["storm_count"] = merged["storm_count"].fillna(0)

    return merged


def train_random_forest(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def visualize_results(X_test, y_test, predictions):
    """Create a 3D scatter plot of true vs predicted weather delays."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color="blue", label="True Delay", s=50)
    ax.scatter(X_test[:, 0], X_test[:, 1], predictions, color="red", marker="^", label="Predicted Delay", s=70)

    ax.set_xlabel("Precipitation (inches)")
    ax.set_ylabel("Storm Count")
    ax.set_zlabel("Weather Delay (minutes)")
    ax.set_title("Random Forest: 3D Predicted vs True Weather Delay")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main(plot_results=True):
    """Main function to train and evaluate a Random Forest model."""
    merged_data = load_and_clean_data()

    X = merged_data[["precipitation", "storm_count"]].values
    y = merged_data["weather_delay"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_random_forest(X_train, y_train)
    predictions = model.predict(X_test)

    # Save model (no need to scale for RF, so no scaler)
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/jfk_weather_rf_model.pkl")
    print("Random Forest model saved to ../models/")

    # Evaluation
    rmse = math.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    if plot_results:
        visualize_results(X_test, y_test, predictions)


if __name__ == "__main__":
    main()
