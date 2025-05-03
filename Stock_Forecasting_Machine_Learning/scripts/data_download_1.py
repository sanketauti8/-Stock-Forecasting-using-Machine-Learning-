import time
import datetime
import pandas as pd
import os


period1 = int(time.mktime(datetime.datetime(2019, 10, 1, 23, 59).timetuple()))
period2 = int(time.mktime(datetime.datetime(2022, 10, 31, 23, 59).timetuple()))
interval = '1d' # 1d, 1m


tickers = ['WMT', 'AMZN' , 'AAPL' , 'CVS.F' , 'GOOGL', 'GE', 'IBM', 'MSFT', 'META', 'DELL' , 'NFLX', 'JPM' ]
for ticker in tickers :

    query_string = f'https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={period1}&period2={period2}&interval={interval}&events=history&includeAdjustedClose=true'
    df = pd.read_csv(query_string)
    outname = ticker + '.csv'
    outdir =   "C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Stock_Forecasting\\datasets\\" + ticker

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, outname) 

    df.to_csv(fullname)
