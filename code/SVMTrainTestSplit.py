# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV


def train_model():
    # ----------- Load and Clean JFK Delay Data -----------
    delay_df = pd.read_csv("../data/clean/JFK_Weather_Delay_Data5.csv")
    delay_df["airport"] = delay_df["airport"].str.strip()
    jfk_df = delay_df[delay_df["airport"] == "JFK"].reset_index(drop=True)

    # Group by year + month
    monthly_delay = jfk_df.groupby(["year", "month"])["weather_delay"].sum().reset_index()
    monthly_delay["year"] = monthly_delay["year"].astype(int)
    monthly_delay["month"] = monthly_delay["month"].astype(int)

    # ----------- Clean Precipitation Data -----------
    precip_df = pd.read_csv("../data/raw/raw data/JFKWeatherData/monthly_total_precipitation5.csv", skiprows=1)
    precip_df.columns = precip_df.columns.str.strip()
    precip_df = precip_df[pd.to_numeric(precip_df["Year"], errors="coerce").notna()]
    precip_df["Year"] = precip_df["Year"].astype(int)
    if "Annual" in precip_df.columns:
        precip_df = precip_df.drop(columns=["Annual"])

    precip_long = precip_df.melt(id_vars=["Year"], var_name="month", value_name="precipitation")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    precip_long["month"] = pd.Categorical(precip_long["month"], categories=months, ordered=True)
    month_num = precip_long["month"].cat.codes + 1
    precip_long["month"] = month_num
    precip_long = precip_long.rename(columns={"Year": "year"})
    precip_long["year"] = precip_long["year"].astype(int)

    # ----------- Clean Storm Data -----------
    storm_df = pd.read_csv("../data/raw/raw data/JFKWeatherData/storm_data_search_results5.csv")
    storm_df["BEGIN_DATE"] = pd.to_datetime(storm_df["BEGIN_DATE"], errors="coerce")
    storm_df["year"] = storm_df["BEGIN_DATE"].dt.year
    storm_df["month"] = storm_df["BEGIN_DATE"].dt.month

    # Filter for relevant storm types
    relevant_types = ["High Wind", "Thunderstorm Wind", "Flash Flood", "Winter Weather", "Strong Wind", "Heavy Snow", "Hail"]
    storm_df = storm_df[storm_df["EVENT_TYPE"].isin(relevant_types)]

    # Count storms per month
    storm_counts = storm_df.groupby(["year", "month"]).size().reset_index(name="storm_count")
    storm_counts["year"] = storm_counts["year"].astype(int)
    storm_counts["month"] = storm_counts["month"].astype(int)

    # ----------- Merge All Features Together -----------
    merged = monthly_delay.merge(precip_long, on=["year", "month"], how="left")
    merged = merged.merge(storm_counts, on=["year", "month"], how="left")
    merged["storm_count"] = merged["storm_count"].fillna(0)

    # Create merged features to CSV 
    merged.to_csv("../data/clean/JFK_combined_features.csv", index=False)

    # ----------- Train Model -----------
    X = merged[["precipitation", "storm_count"]]
    y = merged["weather_delay"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model = SVR(kernel="rbf", C=1000, epsilon=10, gamma=0.1)
    # model.fit(X_train_scaled, y_train)
    # predictions = model.predict(X_test_scaled)

    param_grid = {
        'C': [10, 100, 1000],
        'epsilon': [1, 10, 100],
        'gamma': [0.01, 0.1, 1]
    }
    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_search.fit(X_train_scaled, y_train)


    model = grid_search.best_estimator_
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
    ax.set_title("LinearSVR: 3D Predicted vs True Weather Delay")

    # Legend and formatting
    ax.legend()
    plt.tight_layout()
    plt.show()


    return scaler, model

# Run training
train_model()
