import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# Generate sample data for the price graph
np.random.seed(0)
time = np.arange(0, 100, 1)
price = np.sin(0.1 * time) + np.random.normal(0, 0.1, len(time))

# Detect peaks and valleys
peak_indices, _ = find_peaks(price)
valley_indices, _ = find_peaks(-price)

# Plot the price data
plt.figure(figsize=(12, 6))
plt.plot(time, price, label='Price', color='blue')

# Annotate the prophet best strategy
for peak in peak_indices:
    plt.annotate('Ask Order', xy=(peak, price[peak]), xytext=(peak, price[peak] + 0.3),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 horizontalalignment='center', verticalalignment='bottom', color='red')
    plt.plot(peak, price[peak], 'ro')  # Mark the ask order with a red dot

for valley in valley_indices:
    plt.annotate('Bid Order', xy=(valley, price[valley]), xytext=(valley, price[valley] - 0.3),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 horizontalalignment='center', verticalalignment='top', color='green')
    plt.plot(valley, price[valley], 'go')  # Mark the bid order with a green dot

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Prophet Best Strategy: Price Graph with Annotations')
plt.legend()
plt.grid(True)

# Display the plot
plt.show()
