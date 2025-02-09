# CS506 Project: Flight Delay Prediction

## Description
Flight delays have always been a major inconvenience for both travelers and airline companies. Our project aims to develop a predictive model that estimates flight delays specifically for American Airlines based on various factors such as weather conditions, historical flight statistics, and airline schedules. We hope the ability to accurately predict flight delays can help travelers make informed decisions and assist American Airlines in improving flight efficiency.

## Goal
The primary goal of this project is to build a model that can predict American Airlines’ flight delays based on historical flight data and external factors such as weather and flight schedules.

## Data Collection
We will collect data from the following sources:

- OpenSky Network: [https://opensky-network.org/](https://opensky-network.org/)
- Bureau of Transportation Statistics: [https://www.bts.gov/topics/airlines-airports-and-aviation](https://www.bts.gov/topics/airlines-airports-and-aviation)
- Live American Airlines Flight Status: [https://www.flightaware.com/live/fleet/AAL](https://www.flightaware.com/live/fleet/AAL)
- Aviation Weather Center: [https://aviationweather.gov/](https://aviationweather.gov/)
- All Major Airline Stats 2024 (BTS): [https://explore.dot.gov/views/ontime_7_7a/Table7](https://explore.dot.gov/views/ontime_7_7a/Table7)
- Flight Delay & Cancellation Dataset (2019-2023) - Kaggle: [https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data)
- API for Flight Cancellations Until 2023: [https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)

## Visualization
We will use the following visualization techniques:

- **Line Plot**: To visualize the number of flights delayed for each month in the year 2024 for American Airlines. There will be two line plots, one for delays and one for on-time flights, using different colors.
- **Pie Chart**: To visualize the percentage of flights that were delayed versus those that were on time for the year 2024. The chart will also include canceled and diverted flights.

## Test Plan
The model will be tested using historical data and real-time flight updates. The following steps will be taken:

1. Data preprocessing and cleaning to remove inconsistencies.
2. Splitting data into training and testing sets.
3. Implementing various machine learning models to determine the most accurate predictor.
4. Evaluating model performance using appropriate metrics such as accuracy, precision, recall, and F1-score.
5. Comparing predicted delays with actual flight statuses.

## Contributors
- **[Kevin Dong]**
- **[Alex Chen]**
- **[Kevin Tan]**


