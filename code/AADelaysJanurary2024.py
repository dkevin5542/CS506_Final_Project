import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
# Change the path if needed
df = pd.read_csv('/Users/kevintan/Downloads/CS506/Final Project/AADelaysJanurary2024.csv')

# Filter for American Airlines (AA) flights and delays greater than 0
aa_delays = df[(df['OP_UNIQUE_CARRIER'] == 'AA') & (df['ARR_DELAY_NEW'] > 0)]

# Extract day from the FL_DATE column
aa_delays['DAY'] = pd.to_datetime(aa_delays['FL_DATE']).dt.day

# Count the number of delayed flights per day
delays_per_day = aa_delays.groupby('DAY').size()

# Plot the number of delayed flights per day
plt.figure(figsize=(10,5))
plt.plot(delays_per_day.index, delays_per_day.values, marker='o', linestyle='-', color='b')
plt.xlabel('Day of January 2024')
plt.ylabel('Number of Delayed Flights')
plt.title('American Airlines Delayed Flights in January 2024')
plt.xticks(range(1, 32))  # Ensures all days are labeled
plt.grid()
plt.show()
