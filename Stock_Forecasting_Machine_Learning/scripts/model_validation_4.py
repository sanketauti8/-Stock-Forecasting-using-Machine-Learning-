from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
# from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


def _model_training (df, model, lags) :

    max_date = df["Date"].max()
    lag_columns = []
    for i in range (1,lags):
        lag_columns.append('lag_' + str(i))


    # Index(['Date', 'Unnamed: 0', 'Open', 'High', 'Low', 'Close', 'Adj Close',
    #    'Volume', 'Stock_Name', 'day', 'month', 'year', 'lag_1', 'lag_2',
    #    'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_mean', 'lag_sum', 'lag_std'],
    #   dtype='object')

    redundant_cols = ['Unnamed: 0', 'Open', 'Low', 'Close', 'Adj Close', 'Volume', 'Stock_Name']

    df_data = df.drop(redundant_cols, axis=1)



    # col_list  = [col for col in df if col.startswith('lag_')] + ["High"]
    # print (df.columns)
    # df_data = df[ ["Date"] + col_list ]
    # df_data.dropna(inplace = True)
    df_data.reset_index(drop = True,inplace = True)


    df_data.sort_values(by='Date')
    date_split_1 = max_date - pd.DateOffset(120)
    date_split_2 = max_date - pd.DateOffset(60)
    # print ("Max_date", max_date)
    # print ("date_split_split_1", date_split_1)
    # print ("date_split_split_2", date_split_2)

    train_data = df_data[df_data['Date'] <= date_split_1]
    test_data = df_data[(df_data['Date'] > date_split_1) & (df_data['Date'] <= date_split_2)]    


    train_data.dropna(inplace=True)

    # Assigning values to x and y

    drop_cols = ['High',  'Date']
    traincols = train_data.columns
    feature_cols  = list(set(traincols)-set(drop_cols))
    label_col = 'High'

    x = train_data[feature_cols]
    y = train_data[label_col]

    # print(feature_cols)


    #Model Training and parameter tuning using grid search

    if model == "XgBoost":
        model_name = xgb.XGBRegressor()
        print (" Model Training...........")
        model_name.fit(x, y)

    elif model == "Random Forest" :
        model_name = RandomForestRegressor()
        model_name.fit(x, y)

    elif model == "Ridge" : 
        model_name = Ridge()
        model_name.fit(x, y)

    elif model == "Lasso":    
        model_name = Lasso()
        model_name.fit(x, y)

    test_data.reset_index(drop=True,inplace=True)

    testpart = test_data[test_data['Date']== date_split_1 + pd.DateOffset(1)]
    testpart.reset_index(drop=True, inplace = True)
    keyd = testpart.loc[[0], ['Date']]


    testing = testpart

    predicted_xgb =  model_name.predict(testpart[feature_cols])
    print (predicted_xgb)


    lastlag = testpart
    finalpred = keyd.copy()

    finalpred['Actual']=testpart['High'][0]
    finalpred[model]=np.round(list(predicted_xgb)[0], 2)

    finaloutput = pd.DataFrame()

    for num, dat in enumerate (test_data.Date):
          if num != 0 :
            testpart = test_data[test_data['Date']== dat]
            testpart = testpart.reset_index(drop=True)
            # print (testpart)
            
            #updating lags
            testpart[lag_columns[1 : ]] = lastlag[lag_columns[0 : -1]].values
            testpart['lag1']=list(finalpred[model])[-1]

            testpart['lag_mean'] = testpart[lag_columns].mean(axis=1)
            testpart['lag_sum'] = testpart[lag_columns].sum(axis=1)
            testpart['lag_std'] = testpart[lag_columns].std(axis=1)

            testing = testing.append(testpart, ignore_index = True, sort = False)

            predicted_xgb =  model_name.predict(testpart[feature_cols])
            # print (predicted_xgb)

            partpred=keyd.copy()
            partpred['Date']=testpart['Date'][0]
            partpred['Actual']=testpart['High'][0]
            partpred[model]=np.round(list(predicted_xgb)[0],2)
            

            finalpred = finalpred.append(partpred, ignore_index = True, sort = False)
            lastlag = testpart


    finaloutput  = finaloutput.append(finalpred, ignore_index = True, sort = False)
    finaloutput.to_csv('C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Final_Stock_Forecasting\\validation_predictions.csv', index = False, mode = 'a' )
    # testing.to_csv('C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Final_Stock_Forecasting\\validation_testing_predictions.csv', index = False)
    return finaloutput 





























