# CS506 Project: Airport Flight Delay Prediction

## Description
Flight delays have always been a major inconvenience for both travelers and airline companies. Our project aims to develop a predictive model that estimates flight delays for airports based on various factors such as weather conditions, historical flight statistics, and airline schedules. We hope the ability to accurately predict flight delays can help travelers make informed decisions when flying.

## Goal
The primary goal of this project is to build a model that can predict airport delays based on historical flight data and external factors such as weather and flight schedules.

## Data Collection
We will collect data from the following sources:

- OpenSky Network: [https://opensky-network.org/](https://opensky-network.org/)
- Bureau of Transportation Statistics: [https://www.bts.gov/topics/airlines-airports-and-aviation](https://www.bts.gov/topics/airlines-airports-and-aviation)
- Live American Airlines Flight Status: [https://www.flightaware.com/live/fleet/AAL](https://www.flightaware.com/live/fleet/AAL)
- Aviation Weather Center: [https://aviationweather.gov/](https://aviationweather.gov/)
- All Major Airline Stats 2024 (BTS): [https://explore.dot.gov/views/ontime_7_7a/Table7](https://explore.dot.gov/views/ontime_7_7a/Table7)
- Flight Delay & Cancellation Dataset (2019-2023) - Kaggle: [https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data](https://www.kaggle.com/datasets/patrickzel/flight-delay-and-cancellation-dataset-2019-2023/data)
- API for Flight Cancellations Until 2023: [https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr](https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr)
- Airport Statistics: [https://www.panynj.gov/airports/en/statistics-general-info.html](https://www.panynj.gov/airports/en/statistics-general-info.html)
- Airport Delay Stats: [https://www.flightaware.com/live/airport/delays](https://www.flightaware.com/live/airport/delays)

## Data Cleaning
- Filtering by airport.
- Removal of flight cancellations.
- Extracting only useful data such as the difference in minutes between scheduled and actual departure time.
- Discarding irrelevant information.

## Data Modeling
- Supervised learning models such as XGBoost and logistic regression.

## Visualization
- **Line Plot**: To visualize the number of flights delayed for each month in the year 2024 for a specific airport.
- **Pie Chart**: To visualize the percentage of flights delayed vs. flights on time for 2024.
- **Climate Graphs**: To show the average temperature and amount of precipitation during each month.

## Test Plan
1. Process the data to remove inconsistencies (e.g., filter flights to the specific airport).
   - Calculate the total number of flights from the airport.
   - Find the percentage of delayed flights per month.
   - Gather general weather trends for the airport’s location.
2. Split data into training and testing sets (20-30% withheld for testing).
3. Train on past data (Jan - Dec 2024) and test on future data (e.g., March - August 2025).


## Contributors
- **Kevin Dong**
- **Alex Chen**
- **Kevin Tan**


