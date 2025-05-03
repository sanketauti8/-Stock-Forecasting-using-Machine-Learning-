# import the libraries
import pandas as pd 
import numpy as np






def _data_preprocessing (df, stock_name) :
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace = True)
    df["Stock_Name"] = stock_name

    max_date = df["Date"].max() 
    min_date = df["Date"].min()

    #genearate missing_dates and fill NaN values using forward fill method

    # print ("max_date : ",  max_date)
    # print ("min_date : ",  min_date)
    # print ("end : ",  max_date + )
    df_date = pd.DataFrame({'Date':pd.date_range(start=min_date, end=max_date)})
    df_date = df_date.merge(df, on  = 'Date',  how = 'left')
    df_date.fillna({'Stock Name':stock_name}, inplace = True)
    # df_date.fillna(0, inplace = True)
    df_date.ffill(axis = 0, inplace = True)
    return df_date



def _feature_engineering(df, lags):

    # temporal features
    df['day'] = np.cos(df['Date'].dt.day)
    df['month'] = np.cos(df['Date'].dt.month)
    df['year'] = df['Date'].dt.year

    lag_columns = []

 
    #lag features
    for x in range(1,lags):
        lag_columns.append('lag_' + str(x))

        df['lag_' + str(x)] = df.groupby('Stock_Name')['High'].shift(x)


    df['lag_mean'] = df[lag_columns].mean(axis=1)
    df['lag_sum'] = df[lag_columns].sum(axis=1)
    df['lag_std'] = df[lag_columns].std(axis=1)

    df.loc[df['Date'].dt.day_name().isin(['Monday', 'Tuesday', 'Wedenesday', 'Thursday', 'Friday']), 'Market'] = 1
    df.fillna({'Market': 0}, inplace = True)

    return df









