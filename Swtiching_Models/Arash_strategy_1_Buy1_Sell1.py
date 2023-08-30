import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os
from datetime import timedelta

def moving_average_crossover_strategy(data, short_window=1, long_window=400, R=0.008, R2=0.1, allow_shortselling=True):
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    # Create short-term and long-term moving averages
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Generate signals
    buy_signal = (signals['short_mavg'] > signals['long_mavg'] * (1 + R)) | \
                 (signals['short_mavg'] < signals['long_mavg'] * (1 - R2))
    sell_signal = (signals['short_mavg'] < signals['long_mavg'] * (1 - R)) | \
                  (signals['short_mavg'] > signals['long_mavg'] * (1 + R2))

    signals['signal'] = np.where(buy_signal, 1.0, np.where(sell_signal, -1.0, 0.0))

    if not allow_shortselling:
        signals['signal'] = np.where(signals['signal'] == -1, 0, signals['signal'])

    return signals

# Load historical price data from a text file
file_path = "/Volumes/Financial_Data/Futures/futures-active_UNadjusted_1hour_pafnlm 2/MET_continuous_UNadjusted_1hour.txt"
data = pd.read_csv(file_path, header=None, index_col=0, parse_dates=True, sep=',')
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# Define the date range (start_date and end_date)
start_date = '2021-12-05 18:00:00'
end_date = '2022-04-05 18:00:00'

# Select data within the date range
# data = data[start_date:end_date]
data = data[:]

# Define the allow_shortselling variable
allow_shortselling = True  # Set allow_shortselling to True or False as desired

# Run the strategy function
signals = moving_average_crossover_strategy(data, allow_shortselling=allow_shortselling)

# Merge signals with the original data
data = pd.concat([data, signals], axis=1)

# Calculate the total number of shares owned
shares_owned = 0
shares_owned_list = []
for i, row in data.iterrows():
    if row['signal'] == 1 and shares_owned == 0:
        shares_owned = 1
    elif row['signal'] == -1 and shares_owned == 1:
        shares_owned = 0
    shares_owned_list.append(shares_owned)
data['SharesOwned'] = shares_owned_list

# Calculate profit/loss based on the signals
data['Profit'] = 0.0
buy_price = None
for i, row in data.iterrows():
    if row['signal'] == 1 and data['signal'].shift(1).loc[i] != 1:
        buy_price = row['Close']
    elif row['signal'] == -1 and data['signal'].shift(1).loc[i] != -1:
        if buy_price is not None:
            data.at[i, 'Profit'] = row['Close'] - buy_price

# Calculate cumulative profit over time
data['CumulativeProfit'] = data['Profit'].cumsum()

# Calculate TradeResult (1 for all trades with a sale, 0 otherwise)
data['TradeResult'] = np.where((data['SharesOwned'].shift() == 1) & (data['SharesOwned'] == 0), 1, 0)

# Count the number of trades resulting in profit and loss
num_trades = len(data[data['TradeResult'] == 1])
num_wins = len(data[(data['TradeResult'] == 1) & (data['Profit'] > 0)])
num_losses = num_trades - num_wins

# Calculate win percentage
win_percentage = num_wins / num_trades

# Perform hypothesis test
p_value = norm.sf(num_wins - 1, loc=num_trades * 0.5, scale=np.sqrt(num_trades * 0.25))  # One-sided test
#the alternate hypothesis is that the win percentage is greater than 0.5, so
#many successes should result in a small p-value

# Extract the filename from the file_path
file_name = os.path.basename(file_path)
title = f"Moving Average Crossover Strategy - {file_name}"

# Plot the price and signal with moving average
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Figure 1: Price and Signals
ax1.plot(data.index, data['Close'], label='Price')
ax1.plot(data.index, data['short_mavg'], label='Short-term Moving Average')
ax1.plot(data.index, data['long_mavg'], label='Long-term Moving Average')
ax1.scatter(data[data['signal'] == 1].index, data[data['signal'] == 1]['Close'], marker='^', color='g', label='Buy')
ax1.scatter(data[data['signal'] == -1].index, data[data['signal'] == -1]['Close'], marker='v', color='r', label='Sell')
ax1.set_ylabel('Price')
ax1.set_title(title)  # Use the filename as the title
ax1.legend()

# Figure 2: Cumulative Profit
ax2.plot(data.index, data['CumulativeProfit'])
ax2.set_xlabel('Date')
ax2.set_ylabel('Cumulative Profit')
ax2.axhline(0, color='k', linestyle='--')
ax2.set_title('Cumulative Profit over Time')

# Display p-value, total profit, win percentage, and max profit
total_profit = data['CumulativeProfit'].iloc[-1]
max_profit = data['Profit'].max()

ax2.text(0.02, 0.95, f"P-value: {p_value:.4f}", transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
ax2.text(0.02, 0.9, f"Total Profit: ${total_profit:.2f}", transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
ax2.text(0.02, 0.85, f"Win Percentage: {win_percentage:.2%}", transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))
ax2.text(0.02, 0.8, f"Max Profit: ${max_profit:.2f}", transform=ax2.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

plt.tight_layout()
plt.show()

print(num_wins, num_trades)

data.to_csv('output2.csv', index=True)
