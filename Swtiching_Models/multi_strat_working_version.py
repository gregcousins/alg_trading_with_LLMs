import random  # Importing the random library
import pandas as pd
from scipy.stats import binomtest
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import timedelta

start_time = time.time()

file_path = "/Volumes/Financial_Data/Futures/futures-active_UNadjusted_1hour_pafnlm 2/M2K_continuous_UNadjusted_1hour.txt"
column_names = ['Open', 'High', 'Low', 'Close', 'Volume']
data = pd.read_csv(file_path, header=None, index_col=0, parse_dates=True, sep=',', names=column_names)

# Define the date range (start_date and end_date)
start_date = '2021-12-05 18:00:00'
end_date = '2022-02-05 18:00:00'

# Select data within the date range
data = data[:]
#data = data[start_date:end_date]



data['Action'] = 0
data['TradeProfit'] = 0.0
data['WinLoss'] = 0
data['TotalProfit'] = 0.0
data['CurrentStrategy'] = ''
data['SwitchPValue'] = np.nan  # Initialize with NaNs (will remain NaN unless there's a switch)
data['SwitchOccured'] = 0

# Additional variable to keep track of whether we currently own a share
owns_share = False
buy_price = 0.0
test_tolerance = 0.1 #this is the significance of the hypothesis test in the switching mechanism

# Counter for number of sell trades under the current strategy
sell_trade_counter = 0
switch_counter=30

last_p_value = None



def should_switch(data):
    last_switch_counter_sales = data[data['Action'] == -1].tail(switch_counter)
    num_positive_profit = sum(last_switch_counter_sales['WinLoss'] > 0)
    p_value = binomtest(num_positive_profit, n=switch_counter, p=0.5, alternative='less')
    return p_value
    
def crossover_strategy(data):
    if len(data) < 21:  # Ensure we have enough data
        return pd.Series([0] * len(data), index=data.index)
    
    short_window = 5
    long_window = 20
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_mavg'] = data['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_mavg'] = data['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
    
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] > signals['long_mavg'][short_window:], 1.0, 0.0)   
    signals['positions'] = signals['signal'].diff()
    
    return signals['positions']

def bollinger_band_strategy(data):
    if len(data) < 21:  # Ensure we have enough data
        return pd.Series([0] * len(data), index=data.index)
    
    window = 20
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['mavg'] = data['Close'].rolling(window=window, min_periods=1, center=False).mean()
    signals['std_dev'] = data['Close'].rolling(window=window, min_periods=1, center=False).std()
    signals['upper_band'] = signals['mavg'] + (signals['std_dev'] * 2)
    signals['lower_band'] = signals['mavg'] - (signals['std_dev'] * 2)
    
    signals['signal'] = 0.0
    signals['signal'][window:] = np.where(signals['price'][window:] < signals['lower_band'][window:], 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    
    return signals['positions']

def rsi_strategy(data):
    if len(data) < 15:  # Ensure we have enough data
        return pd.Series([0] * len(data), index=data.index)
    
    window = 14  # Commonly used RSI window
    signals = pd.DataFrame(index=data.index)
    
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    signals['rsi'] = 100 - (100 / (1 + rs))
    
    signals['signal'] = 0.0
    signals['signal'][signals['rsi'] < 30] = 1.0
    signals['signal'][signals['rsi'] > 70] = -1.0
    signals['positions'] = signals['signal'].diff()
    
    return signals['positions']
    
def macd_strategy(data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0
    signals['signal'] = np.where(macd > signal, 1.0, 0.0)
    signals['positions'] = signals['signal'].diff()
    
    return signals['positions']
    
def parabolic_sar_strategy(data):
    # Using a simple implementation of Parabolic SAR without acceleration factor adjustments
    start, increment, maximum = 0.02, 0.02, 0.2
    long = True
    sar = [data['Low'][0]]
    ep = data['High'][0]
    af = start
    signals = pd.DataFrame(index=data.index)
    signals['signal'] = 0.0

    for i in range(1, len(data)):
        temp_sar = sar[-1] + af * (ep - sar[-1])
        
        if long:
            sar.append(min(temp_sar, data['Low'][i-1], data['Low'][i]))
            if data['Close'][i] < sar[-1]:
                long = False
                sar[-1] = max(data['High'][i-1], data['High'][i])
                ep = data['Low'][i]
                af = start
        else:
            sar.append(max(temp_sar, data['High'][i-1], data['High'][i]))
            if data['Close'][i] > sar[-1]:
                long = True
                sar[-1] = min(data['Low'][i-1], data['Low'][i])
                ep = data['High'][i]
                af = start
        
        if long and data['High'][i] > ep:
            ep = data['High'][i]
            af = min(af + increment, maximum)
        elif not long and data['Low'][i] < ep:
            ep = data['Low'][i]
            af = min(af + increment, maximum)
        
        if (long and data['Close'][i] > sar[-1]) or (not long and data['Close'][i] < sar[-1]):
            signals.at[data.index[i], 'signal'] = 1.0 if long else 0.0
        else:
            signals.at[data.index[i], 'signal'] = 0.0
    
    signals['positions'] = signals['signal'].diff()
    
    return signals['positions']


# List of strategies
strategies = [crossover_strategy, bollinger_band_strategy, rsi_strategy,macd_strategy, parabolic_sar_strategy]  # Add more strategies here when you have them

# Initializing the current strategy
current_strategy = random.choice(strategies)  # You can choose a strategy at random initially or assign one directly


#main loop



for i in range(len(data)):
    # ... [Rest of your loop logic]
    action_series = current_strategy(data.iloc[:i + 1])
    if not action_series.empty:
        data.loc[data.index[i], 'Action'] = action_series.iloc[-1]
        
    # Check if the action for this timestamp is a sell
    if data.loc[data.index[i], 'Action'] == -1:
        sell_trade_counter += 1  # Increment the sell trade counter

    if sell_trade_counter >= switch_counter:
        p_value_current = should_switch(data).pvalue
        if p_value_current < test_tolerance and last_p_value >= test_tolerance:  
            # Only mark a switch when p-value goes below threshold for the first time
            # Mark that a switch occurred for plotting
            data.loc[data.index[i], 'SwitchOccured'] = 1
            # Selecting a new strategy, excluding the current one
            available_strategies = [s for s in strategies if s != current_strategy]
            current_strategy = random.choice(available_strategies)
            sell_trade_counter = 0
        last_p_value = p_value_current

    # If we are within switch_counter steps from the start or a switch, use the last p-value
    elif i < switch_counter or (i - sell_trade_counter) < switch_counter:
        p_value_current = last_p_value
    else:
        # Calculate the current p-value without switching
        p_value_current = should_switch(data).pvalue

    # Record the p-value at each step
    data.loc[data.index[i], 'SwitchPValue'] = p_value_current
    # Record the current strategy being used
    data.loc[data.index[i], 'CurrentStrategy'] = current_strategy.__name__

    action_series = current_strategy(data.iloc[:i + 1])

    # No action if action_series is empty
    if action_series.empty:
        data.loc[data.index[i], 'Action'] = 0
        continue
    
    # Get the latest action signal
    current_action = action_series.iloc[-1]

    # Ensure we can't have a sell action if we don't own a share
    if current_action == -1 and not owns_share:
        current_action = 0

    # If we are to buy, we need to ensure we don't already own a share
    if current_action == 1 and owns_share:
        current_action = 0

    # Update the 'Action' column with the current action
    data.loc[data.index[i], 'Action'] = current_action

    # If current action is to buy, update owns_share and buy_price
    if current_action == 1:
        owns_share = True
        buy_price = data.loc[data.index[i], 'Close']
    # If current action is to sell, calculate profit and update owns_share
    elif current_action == -1:
        trade_profit = data.loc[data.index[i], 'Close'] - buy_price
        data.loc[data.index[i], 'TradeProfit'] = trade_profit
        data.loc[data.index[i], 'WinLoss'] = 1 if trade_profit > 0 else -1
        owns_share = False


# Initialize total profit column
data['TotalProfit'] = data['TradeProfit'].cumsum().fillna(0)


# Save to CSV
data.to_csv('output_with_new_features.csv')


file_name = os.path.basename(file_path)
title = f"Total Profit/Loss for Two Strategy Switch - {file_name}"

import plotly.graph_objects as go

# Downsample or aggregate the data for efficient plotting
resampled_data = data.resample('D').last()  # Aggregating the data daily. You can adjust this.

# Color for each strategy to visually identify during switches
strategy_colors = {
    "crossover_strategy": "blue",
    "bollinger_band_strategy": "green",
    "rsi_strategy": "yellow",
    "macd_strategy": "purple",
    "parabolic_sar_strategy": "orange"
}

# Plot
fig = go.Figure()

# Plot Profit/Loss Over Time
fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data['TotalProfit'],
                    mode='lines',
                    name='Profit/Loss'))

# Plot Closing Prices
fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data['Close'],
                    mode='lines',
                    name='Closing Price'))

# Indicate where a strategy switch occurs
switch_data = data[data['SwitchOccured'] == 1]
y_min = min(resampled_data['Close'].min(), resampled_data['TotalProfit'].min())
y_max = max(resampled_data['Close'].max(), resampled_data['TotalProfit'].max())

for _, switch_row in switch_data.iterrows():
    strategy_name = switch_row['CurrentStrategy']
    strategy_color = strategy_colors.get(strategy_name, "black")  # Default to black if strategy name is not in dictionary
    fig.add_shape(
        dict(
            type="line",
            x0=switch_row.name,
            x1=switch_row.name,
            y0=y_min,
            y1=y_max,
            line=dict(
                color=strategy_color,
                width=2,  # Make the line thicker
                dash="dot",
            )
    ))


# Add legends for the strategies
for strategy_name, strategy_color in strategy_colors.items():
    fig.add_trace(
        go.Scatter(x=[None], y=[None], mode='lines', 
                   line=dict(color=strategy_color, width=0.5, dash="dot"),
                   name=strategy_name)
    )


# Add the Switching P-Value
fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data['SwitchPValue'],
                    mode='lines',
                    name='Switching P-Value',
                    line=dict(width=0.5, dash='dot', color='red'),
                    yaxis="y2"))

# Adding the final profit and p-value as annotations using resampled_data
final_profit = resampled_data['TotalProfit'].iloc[-1]
final_p_value = resampled_data['SwitchPValue'].iloc[-1]

fig.add_annotation(
    x=resampled_data.index[-1],
    y=final_profit,
    xanchor='right',
    yanchor='bottom',
    text=f"Final Profit/Loss: {final_profit:.2f}",
    showarrow=False,
    font=dict(color="blue")
)

fig.add_annotation(
    x=resampled_data.index[-1],
    y=final_p_value,
    xanchor='right',
    yanchor='top',
    text=f"Final P-Value: {final_p_value:.4f}",
    showarrow=False,
    yshift=40,  # To avoid overlap with the profit annotation
    font=dict(color="red")
)

fig.update_layout(
    title=f"Total Profit/Loss for Two Strategy Switch - {file_name}",
    xaxis_title='Date',
    yaxis_title='Profit/Loss Over Time',
    yaxis2=dict(title='Switching P-Value', overlaying='y', side='right'),
    template="plotly_dark"
)

fig.show()

fig.write_html("output_plot.html")



end_time = time.time()
runtime = end_time - start_time
print("Runtime: {:.4f} seconds".format(runtime))