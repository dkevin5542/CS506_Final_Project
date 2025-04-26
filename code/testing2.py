import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load model and scaler
model = joblib.load("../models/rf_weather_model.pkl")
scaler = joblib.load("../models/rf_scaler.pkl")

# --- Load 2024 weather delay data ---
delay_df = pd.read_csv("../data/clean/JFK_Weather_Delay_Data.csv")
delay_df["airport"] = delay_df["airport"].str.strip()

# --- Group by month ---
monthly_delay = delay_df.groupby(["year", "month"])["weather_delay"].sum().reset_index()

# --- Load and clean precipitation data ---
precip_df = pd.read_csv("../data/raw/raw data/JFKWeatherData/monthly_total_precipitation21.csv", skiprows=2)
precip_df.columns = precip_df.columns.str.strip()
if precip_df.columns[0] != 'Year':
    precip_df = precip_df.drop(precip_df.columns[0], axis=1)
precip_df = precip_df[pd.to_numeric(precip_df["Year"], errors="coerce").notna()]
precip_df["Year"] = precip_df["Year"].astype(int)
if "Annual" in precip_df.columns:
    precip_df = precip_df.drop(columns=["Annual"])

# Reshape precipitation to long format
precip_long = precip_df.melt(id_vars=["Year"], var_name="month", value_name="precipitation")
precip_long["precipitation"] = pd.to_numeric(precip_long["precipitation"], errors="coerce")
precip_long = precip_long.dropna(subset=["precipitation"])
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
precip_long["month"] = pd.Categorical(precip_long["month"], categories=months, ordered=True)
precip_long["month"] = precip_long["month"].cat.codes + 1
precip_long = precip_long.rename(columns={"Year": "year"})

# Only 2024 precipitation
precip_2024 = precip_long[precip_long["year"] == 2024]

# --- Load and clean storm data ---
storm_df = pd.read_csv(
    "../data/raw/raw data/JFKWeatherData/storm_data_search_results.csv",
    on_bad_lines='skip',
    quoting=1,
    quotechar='"',
    encoding='utf-8',
    dtype=str
)
storm_df["BEGIN_DATE"] = pd.to_datetime(storm_df["BEGIN_DATE"], errors="coerce")
storm_df["year"] = storm_df["BEGIN_DATE"].dt.year
storm_df["month"] = storm_df["BEGIN_DATE"].dt.month
relevant_types = ["Blizzard", "Coastal Flood", "Dense Fog", "Flash Flood", "Flood",
                  "Hail", "Heavy Rain", "Heavy Snow", "High Wind", "Hurricane (Typhoon)",
                  "Ice Storm", "Lightning", "Strong Wind", "Thunderstorm Wind", "Tornado",
                  "Tropical Depression", "Tropical Storm", "Winter Storm", "Winter Weather",
                  "Excessive Heat", "Extreme Cold/Wind Chill", "Storm Surge/Tide", "Wildfire"]
storm_df = storm_df[storm_df["EVENT_TYPE"].isin(relevant_types)]
storm_counts = storm_df.groupby(["year", "month"]).size().reset_index(name="storm_count")
storm_counts["year"] = storm_counts["year"].astype(int)
storm_counts["month"] = storm_counts["month"].astype(int)

# Only 2024 storm counts
storm_2024 = storm_counts[storm_counts["year"] == 2024]


# --- Merge all 2024 features ---
merged_2024 = monthly_delay.merge(precip_2024, on=["year", "month"], how="left")
merged_2024 = merged_2024.merge(storm_2024, on=["year", "month"], how="left")
merged_2024["storm_count"] = merged_2024["storm_count"].fillna(0)

merged_2024.to_csv("../data/clean/JFK_combined_features2024.csv", index=False)

# --- Prepare features and labels ---
X_test = merged_2024[["precipitation", "storm_count"]]
y_true = merged_2024["weather_delay"]

# Scale features
X_test_scaled = scaler.transform(X_test)

# Predict
y_pred = model.predict(X_test_scaled)

# --- Create comparison DataFrame ---
comparison_df = pd.DataFrame({
    "Month": merged_2024["month"],
    "Actual_Delay": y_true,
    "Predicted_Delay": y_pred
}).sort_values("Month")

print(comparison_df)

# --- Calculate Metrics ---
rmse = math.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# --- Plot ---
plt.figure(figsize=(14, 6))
plt.plot(comparison_df["Month"], comparison_df["Actual_Delay"], marker="o", label="Actual Weather Delay", color="blue")
plt.plot(comparison_df["Month"], comparison_df["Predicted_Delay"], marker="^", label="Predicted Weather Delay", color="red")
plt.xlabel("Month")
plt.ylabel("Delay (minutes)")
plt.title(f"2024 JFK Weather Delay: Actual vs Predicted\nRMSE = {rmse:.2f}, MAE = {mae:.2f}")
plt.legend()
plt.grid(True)
plt.show()
