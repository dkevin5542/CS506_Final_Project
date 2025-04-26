# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import joblib
import os




def train_model():
    # ----------- Load and Clean JFK Delay Data -----------
    delay_df = pd.read_csv("../data/clean/JFK_Weather_Delay_Data21.csv")
    delay_df["airport"] = delay_df["airport"].str.strip()
    jfk_df = delay_df[delay_df["airport"] == "JFK"].reset_index(drop=True)

    # Group by year + month
    monthly_delay = jfk_df.groupby(["year", "month"])["weather_delay"].sum().reset_index()
    monthly_delay["year"] = monthly_delay["year"].astype(int)
    monthly_delay["month"] = monthly_delay["month"].astype(int)

    # ----------- Clean Precipitation Data -----------
    # Load precipitation file
    precip_df = pd.read_csv("../data/raw/raw data/JFKWeatherData/monthly_total_precipitation21.csv", skiprows=2)  # skip first 2 junk rows

    # Fix columns
    precip_df.columns = precip_df.columns.str.strip()

    # Drop the first unnamed column if it exists
    if precip_df.columns[0] != 'Year':
        precip_df = precip_df.drop(precip_df.columns[0], axis=1)

    # Drop any summary rows (Mean, Max, Min)
    precip_df = precip_df[pd.to_numeric(precip_df["Year"], errors="coerce").notna()]
    precip_df["Year"] = precip_df["Year"].astype(int)

    # Drop Annual if present
    if "Annual" in precip_df.columns:
        precip_df = precip_df.drop(columns=["Annual"])

    # Reshape wide to long format
    precip_long = precip_df.melt(id_vars=["Year"], var_name="month", value_name="precipitation")
    precip_long["precipitation"] = pd.to_numeric(precip_long["precipitation"], errors="coerce")
    precip_long = precip_long.dropna(subset=["precipitation"])

    # Map month names correctly
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    precip_long["month"] = pd.Categorical(precip_long["month"], categories=months, ordered=True)
    month_num = precip_long["month"].cat.codes + 1
    precip_long["month"] = month_num

    # Rename 'Year' to 'year' and ensure types
    precip_long = precip_long.rename(columns={"Year": "year"})
    precip_long["year"] = precip_long["year"].astype(int)

    # ----------- Clean Storm Data -----------
    # storm_df = pd.read_csv("../data/raw/raw data/JFKWeatherData/storm_data_search_results21.csv")
    storm_df = pd.read_csv (
        "../data/raw/raw data/JFKWeatherData/storm_data_search_results21.csv",
        on_bad_lines='skip',
        quoting=1,
        quotechar='"',
        encoding='utf-8',
        dtype=str
    )

    storm_df["BEGIN_DATE"] = pd.to_datetime(storm_df["BEGIN_DATE"], errors="coerce")
    storm_df["year"] = storm_df["BEGIN_DATE"].dt.year
    storm_df["month"] = storm_df["BEGIN_DATE"].dt.month

    # Filter for relevant storm types
    relevant_types = ["Blizzard", "Coastal Flood", "Dense Fog", "Flash Flood", "Flood",
                        "Hail", "Heavy Rain", "Heavy Snow", "High Wind", "Hurricane (Typhoon)",
                        "Ice Storm", "Lightning", "Strong Wind", "Thunderstorm Wind", "Tornado",
                        "Tropical Depression", "Tropical Storm", "Winter Storm", "Winter Weather",
                        "Excessive Heat", "Extreme Cold/Wind Chill", "Storm Surge/Tide", "Wildfire"]
    storm_df = storm_df[storm_df["EVENT_TYPE"].isin(relevant_types)]

    # Count storms per month
    storm_counts = storm_df.groupby(["year", "month"]).size().reset_index(name="storm_count")
    storm_counts["year"] = storm_counts["year"].astype(int)
    storm_counts["month"] = storm_counts["month"].astype(int)

    # ----------- Merge All Features Together -----------
    merged = monthly_delay.merge(precip_long, on=["year", "month"], how="left")
    merged = merged.merge(storm_counts, on=["year", "month"], how="left")
    merged["storm_count"] = merged["storm_count"].fillna(0)

    merged = merged[merged["year"] < 2024]

    # Create merged features to CSV 
    merged.to_csv("../data/clean/JFK_combined_features.csv", index=False)

    # ----------- Train Model -----------
    X = merged[["precipitation", "storm_count"]]
    y = merged["weather_delay"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # model = KNeighborsRegressor(n_neighbors=3)  # 3, 5, or 7
    # model.fit(X_train_scaled, y_train)
    # predictions = model.predict(X_test_scaled)

    model = RandomForestRegressor(max_depth=1000)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)


    # # ----------- Visualize Results -----------
    # plt.figure(figsize=(10, 5))
    # plt.scatter(X_test["precipitation"], y_test, label="True", color="blue")
    # plt.scatter(X_test["precipitation"], predictions, label="Predicted", color="red", marker="^")
    # plt.xlabel("Monthly Precipitation (inches)")
    # plt.ylabel("Weather Delay (minutes)")
    # plt.title("LinearSVR: Predicted vs True Weather Delays")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # ----------- 3D Plot: True vs Predicted Weather Delay -----------

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot true values (blue dots)
    ax.scatter(
        X_test["precipitation"],
        X_test["storm_count"],
        y_test,
        color="blue",
        label="True Delay",
        s=50
    )

    # Plot predicted values (red triangles)
    ax.scatter(
        X_test["precipitation"],
        X_test["storm_count"],
        predictions,
        color="red",
        marker="^",
        label="Predicted Delay",
        s=70
    )

    # Axis labels and title
    ax.set_xlabel("Precipitation (inches)")
    ax.set_ylabel("Storm Count")
    ax.set_zlabel("Weather Delay (minutes)")
    ax.set_title("3D Predicted vs True Weather Delay")

    # Legend and formatting
    ax.legend()
    plt.tight_layout()
    plt.show()

    os.makedirs("../models", exist_ok=True)
    joblib.dump(scaler, "../models/rf_scaler.pkl")
    joblib.dump(model, "../models/rf_weather_model.pkl")

    return scaler, model

# Run training
train_model()
