from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
# from sklearn.model_selection import GridSearchCV
import pandas as pd


def _model_prediction_blind_run (df, model, lags) :


    print (model)

    max_date = df["Date"].max()
    lag_columns = []
    for i in range (1,lags):
        lag_columns.append('lag_' + str(i))



    redundant_cols = ['Unnamed: 0', 'Open', 'Low', 'Close', 'Adj Close', 'Volume', 'Stock_Name']

    df_data = df.drop(redundant_cols, axis=1)

    df_data.reset_index(drop = True,inplace = True)


    df_data.sort_values(by='Date')
    date_split = max_date - pd.DateOffset(60)

    print (date_split)

    train_data = df_data[df_data['Date'] <= date_split]
    test_data = df_data[df_data['Date'] > date_split]    


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

    testpart = test_data[test_data['Date']== date_split + pd.DateOffset(1)]
    testpart.reset_index(drop=True, inplace = True)
    keyd = testpart.loc[[0], ['Date']]

    testing = testpart

    predicted_xgb =  model_name.predict(testpart[feature_cols])


    lastlag = testpart
    finalpred = keyd.copy()

    # finalpred['Actual']=testpart['High'][0]
    finalpred[model]=list(predicted_xgb)[0]

    finaloutput = pd.DataFrame()

    for num, dat in enumerate (test_data.Date):
          if num != 0 :
            testpart = test_data[test_data['Date']== dat]
            testpart = testpart.reset_index(drop=True)
            # print (testpart)
            
            #updating lags
            testpart[lag_columns[1 : ]] = lastlag[lag_columns[0 : -1]].values
            testpart['lag_1']=list(finalpred[model])[-1]

            testpart['lag_mean'] = testpart[lag_columns].mean(axis=1)
            testpart['lag_sum'] = testpart[lag_columns].sum(axis=1)
            testpart['lag_std'] = testpart[lag_columns].std(axis=1)

            testing = testing.append(testpart, ignore_index = True, sort = False)

            predicted_xgb =  model_name.predict(testpart[feature_cols])
            # print (predicted_xgb)

            partpred=keyd.copy()
            partpred['Date']=testpart['Date'][0]
            # partpred['Actual']=testpart['High'][0]
            partpred[model]=list(predicted_xgb)[0]
            

            finalpred = finalpred.append(partpred, ignore_index = True, sort = False)
            lastlag = testpart


    finaloutput  = finaloutput.append(finalpred, ignore_index = True, sort = False)
    finaloutput.rename(columns={model : 'Best_Model'}, inplace=True)

    
    finaloutput.to_json('C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Final_Stock_Forecasting\\final_predictions.json')
    # testing.to_csv('C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Final_Stock_Forecasting\\testing_predictions.csv', index = False)
    return finaloutput 





























