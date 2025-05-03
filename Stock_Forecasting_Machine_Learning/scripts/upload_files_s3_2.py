import boto3
s3 = boto3.resource('s3')


tickers = ['WMT', 'AMZN' , 'AAPL' , 'CVS.F' , 'GOOGL', 'GE', 'IBM', 'MSFT', 'META', 'DELL' , 'NFLX', 'JPM' ]


for com in tickers :


    s3.meta.client.upload_file('C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Stock_Forecasting\\datasets\\' + com + '\\' + com + '.csv',
 'fortunestockdata',com + '/' + com + '.csv')