import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def create_feature_dataframe(path, feature_name):
    """ 
    Reads in a csv file with index, commodity, currency data and selects only the date and price columns. Renames the price 
    column based on user feature_name paramaters.

    Parameters:
    path (string): a relative path to the csv file. 'DataSets/Economy/Monthly/canada_cpi_jan_2014_apr_2024_formatted.csv'
    feature_name (string): the name of the column you want to rename from the "price"

    Returns:
    DataFrame: dataframe of the pricing data with date and price of feature
    """
    df = pd.read_csv(path)
    df = df[['Date', 'Price']]
    df.rename(columns={'Price': f'{feature_name}'}, inplace=True)
    return df

def remove_commas(x):
    """ 
    Removes commas from numerical values represented as strings and returns the string as a float.

    Parameters:
    x (string): string numerical value with commas in it

    Returns:
    float value of number
    """
    return float(x.replace(',', '')) if isinstance(x, str) else x

def parse_date(date_str):
    """ 
    Takes a string date and returns a date time object if the date is in a certain format. Mainly to be used when transposing
    monthly and quarterly economic data.

    Parameters:
    date_str (string): In the format:
     - 23-Jun
     - Jun-23
     - 2023-06-30

    Returns:
    datetime (object)
    """
    for fmt in ('%y-%b', '%b-%y','%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    raise ValueError(f"no valid date format found for {date_str}")

def statscan_transpose(csv_path):
    """ 
    The function reads in a csv file of statistics canada economic data and outputs a transposed dataframe with any
    commas removed. The csv file must already have have the header and footer information removed with just the table
    information remaining. All statistics canada tabular data runs horizontally instead of vertically and needs to be
    transposed. This function accomploshises this.

    Parameters:
    csv_path (string): a relative path to the csv file. 'DataSets/Economy/Monthly/canada_cpi_jan_2014_apr_2024_formatted.csv'

    Returns:
    DataFrame: transposed dataframe of the economic data
    """
    #Transpose the table and put the first row as the column headers
    df = pd.read_csv(csv_path)
    df_transpose = df.T.reset_index()
    df_transpose.columns = df_transpose.iloc[0]
    df_transpose = df_transpose[1:]

    #change the string date to a date time object offset to the end of the month
    column_name = df_transpose.columns[0]
    df_transpose.rename(columns={column_name: 'Date'}, inplace=True)
    df_transpose['Date'] = df_transpose['Date'].apply(parseDate)
    df_transpose['Date'] = df_transpose['Date'] + pd.offsets.MonthEnd(1)

    #loop through all columns (except for date) in the df and remove any commas in the data and convert string data type to float
    for col in df_transpose.columns:
        if col != 'Date':
            df_transpose[col] = df_transpose[col].apply(removeCommas)
    
    return df_transpose

def calculate_rsi(prices, n=14):
    """ 
    The Relative Strength Index (RSI) is a momentum oscillator used in technical analysis that measures the speed and 
    change of price movements. It is typically used to identify overbought or oversold conditions in a market. 
    The RSI oscillates between 0 and 100, making it easy to interpret and apply in various trading strategies.

    The function takes in a list of stock prices and calculates the average gain is the sum of all gains over the past n 
    periods divided by n. Similarly, the average loss is the sum of all losses over the past n periods divided by n.

    Parameters:
    prices (series): a series of stock prices

    Returns:
    float: a single RSI value between 0 and 100
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=n, min_periods=1).mean()
    avg_loss = loss.rolling(window=n, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def plot_line_chart(time, data, label):
    """
    Plots a time series feature with the provided time index and data.
    
    Paramters:
    time(Pandas Series or DataFrame index): time index
    data (Series): feature that you wish to plot
    label (String) The label for the plot line (e.g., 'TSX Index Price').
    
    Returns:
    Matplotlib plot line chart over time with dimension 20 x 10 for a given feature
    """
    # Plot the line chart
    plt.plot(time, data, label=label, color='blue')

    # Add titles and labels
    plt.title(f'{data.name} Price Over Time')
    plt.xlabel('Year')
    plt.ylabel('Price')

   # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Optionally add a grid, legend, and format the x-axis for better readability
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)

    # Show the plot
    plt.show()

    plt.figure(figsize=(20, 10))  # Set the figure size

  

def trade_DCA(open_price, close_price, index, df, shares, investment, prediction, expected_return):
    """ 
    Takes the opening and closing prices of the TSX index and determines whether to buy or sell shares and the resulting
    cummulative investment.

    Trading Strategy
    - If the model predicts the index will go down - Purchase one share
    - If the model predicts the index will go up:
        - Sell all shares if the shares held have an average price below the current close price
        - Hold if no shares held or average price is above current close price

    Parameters:
    open (float): TSX closing price from the prior day
    close (float): TSX closing price on the same day
    shares (int): the total number of shares held 
    investment (float): Cummulative investment for the test dataset
    prediction (boolean): y_pred from the XGBoost model
    expected_return (float): expected return on the investment in decimal form 0.05 = 5%

    Returns:
    investment (float): updated cummulative investment value
    shares (int): updated cummulative number of shares
    action (string): Buy, Sell or Hold depending on the below logic
    get_avg_purchase_price (float): cummulative average purchase price of all shares purchased
    """
    # If the index goes up Sell all shares if the average cost (plus a return) is less than the closing price
    if prediction and (shares > 0) and (get_avg_purchase_price(df) * (1 + expected_return) < close_price):     
        proceeds = shares * close_price 
        investment = investment + proceeds - 10  # Subtract transaction cost
        shares = 0
        action = 'sell'
        return investment, shares, action, get_avg_purchase_price(df)
    
    # If the index goes down buy one share
    elif not prediction:     
        if investment > 0:
            investment = investment - open_price - 10 # Subtract transaction cost
            shares += 1
            action = 'buy'
            return investment, shares, action, get_avg_purchase_price(df)
        # if the investment is zero do nothing
        else:
            action = 'hold'
            return investment, shares, action, get_avg_purchase_price(df)
    else:
        action = 'hold'
        return investment, shares, action, get_avg_purchase_price(df) 
    
def trade_purchase(open_price, close_price, index, df, shares, investment, prediction):
    """ 
    Takes the opening and closing prices of the TSX index and determines whether to buy or sell shares and the resulting
    cummulative investment.

    Trading Strategy
    - If the model predicts the index will go down - do nothing
    - If the model predicts the index will go up:
        - If we already hold shares and the average cost is below the open price sell all the shares
        - Use the entire investment to purchase all shares at the open price
        - If the close price is greater than the open price we purchased at, sell all the shares
        - If the close price is less than the open price hold the shares to another day

    Parameters:
    open (float): TSX closing price from the prior day
    close (float): TSX closing price on the same day
    shares (int): the total number of shares held 
    investment (float): Cummulative investment for the test dataset
    prediction (boolean): y_pred from the XGBoost model

    Returns:
    investment (float): updated cummulative investment value
    shares (int): updated cummulative number of shares
    action (string): Buy, Sell or Hold depending on the below logic
    get_avg_purchase_price (float): cummulative average purchase price of all shares purchase
    """
    # if index is predicted to go up and we already hold shares and the average price is less than open, sell all shares
    if prediction and open_price > get_avg_purchase_price(df) and (shares > 0):
        proceeds = shares * open_price
        shares = 0
        investment = investment + proceeds - 10  # Subtract transaction cost
        action = 'sell'
        
        return investment, shares, action, 0
    
    if prediction and investment > 0:
        #If the model predicts the market will go up by at the open use all the investment to buy shares
        additional_shares = investment // open_price

        # Update total shares
        shares = shares + additional_shares

        # Calculate the total cost of the additional shares purchased
        cost_of_new_shares = additional_shares * open_price

        # Deduct the cost from the investment, including any additional fixed cost (like fees)
        investment = investment - cost_of_new_shares - 10

        action = 'buy'
 
    # Closing price is greater than price we purchased at open so sell.
    # Average price is zero since we do not hold any more shares
    if prediction and (close_price > open_price):
        proceeds = shares * close_price 
        shares = 0
        investment = investment + proceeds - 10  # Subtract transaction cost
        action = 'sell'
        return investment, shares, action, 0
    else:
        action = 'hold'
        return investment, shares, action, get_avg_purchase_price(df) 

def get_avg_purchase_price(df):
    """ 
    Calculates the cummulative average of all bought shares.

    Parameters:
    df (Pandas DataFrame): Investment DataFrame must already have a 'Buy' column with purchase prices.

    Returns:
    avg_price (float): updated cummulative investment value
    """

    df['Buy_Shifted'] = df['Buy'].shift(1)

    # Filter out the rows where the 'Buy_Shifted' is not zero
    df_non_zero = df[df['Buy_Shifted'] != 0]

    # Calculate the average price of the prior day's non-zero values
    avg_price = df_non_zero['Buy_Shifted'].mean()

    return avg_price