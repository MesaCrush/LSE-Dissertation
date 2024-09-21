import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt

# Load the data
print("Loading data...")
data = pd.read_csv('data/data_filled.csv')

# Convert 'ts_recv' to datetime format
data['ts_recv'] = pd.to_datetime(data['ts_recv'])

# Calculate the mid price
data['mid_price'] = (data['bid_px_00'] + data['ask_px_00']) / 2

# Check the column names in the DataFrame
print(data.columns)

# Use the correct column name for 'timestamp_column'
unique_timestamps = data['ts_recv'].unique()
num_steps = len(unique_timestamps)
print(num_steps)

# Visualize the mid price with real timestamps
print("\nVisualizing the mid price...")
plt.figure(figsize=(10, 6))
plt.plot(data['ts_recv'], data['mid_price'], label='Mid Price')
plt.xlabel('Timestamp')
plt.ylabel('Mid Price')
plt.title('Mid Price Over Time')
plt.legend()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()