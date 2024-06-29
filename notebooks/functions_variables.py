import pandas as pd
from datetime import datetime

def createFeatureDataFrame(path, feature_name):
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

def removeCommas(x):
    """ 
    Removes commas from numerical values represented as strings and returns the string as a float.

    Parameters:
    x (string): string numerical value with commas in it

    Returns:
    float value of number
    """
    return float(x.replace(',', '')) if isinstance(x, str) else x

def parseDate(date_str):
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

def statscanTranspose(csv_path):
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