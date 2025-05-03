import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=ConvergenceWarning)

from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning

# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# warnings.simplefilter(action="ignore", category=ConvergenceWarning)
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt
import boto3
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
import matplotlib.dates as mdates




from reading_files_s3_3 import _data_preprocessing,_feature_engineering
from model_validation_4 import _model_training
from model_blind_run_5 import _model_prediction_blind_run

def _models (df, lags) :
    mdl_list = ["XgBoost", "Random Forest", "Lasso", "Ridge"]

    for mdl in mdl_list :
        
        res = _model_training(df, mdl, lags)
        if mdl == "XgBoost":
            newres = res
        else :
            newres[mdl] = res[mdl]
    return newres


def _accuracy (df, stock_name) :
    # calculate RMSE


    df['day_of_week'] = df['Date'].dt.day_name()

    # print (df.head(2))
    # print (len(df))
    df = df[~df["day_of_week"].isin(["Saturday", "Sunday"])]
    # print (len(df))


    mdl_list = ["XgBoost", "Random Forest", "Lasso", "Ridge"]



    RMSE_xgb = np.around((mean_squared_error(df["Actual"], df["XgBoost"], squared=False)),2)
    RMSE_rf = np.around((mean_squared_error(df["Actual"], df["Random Forest"], squared=False)),2)
    RMSE_lasso =  np.around((mean_squared_error(df["Actual"], df["Lasso"], squared=False)),2)
    RMSE_ridge =  np.around((mean_squared_error(df["Actual"], df["Ridge"], squared=False)),2)

    MAE_xgb =  np.around((mean_absolute_error(df["Actual"], df["XgBoost"])),2)
    MAE_rf =  np.around((mean_absolute_error(df["Actual"], df["Random Forest"])),2)
    MAE_lasso =  np.around((mean_absolute_error(df["Actual"], df["Lasso"])),2)
    MAE_ridge =  np.around((mean_absolute_error(df["Actual"], df["Ridge"])),2)

    
    # FB_xgb =  np.around( (df["XgBoost"].sum()/df["Actual"]) - 1,2)
    # FB_rf =  np.around( (df["Random Forest"].sum()/df["Actual"]) - 1,2)
    # FB_lasso =  np.around( (df["Lasso"].sum()/df["Actual"]) - 1,2)
    # FB_ridge =  np.around( (df["Ridge"].sum()/df["Actual"]) - 1,2)



    MAPE_xgb = np.around((mean_absolute_percentage_error(df["Actual"], df["XgBoost"]) * 100),2)
    MAPE_rf = np.around((mean_absolute_percentage_error(df["Actual"], df["Random Forest"]) * 100),2)
    MAPE_lasso = np.around((mean_absolute_percentage_error(df["Actual"], df["Lasso"]) * 100),2)
    MAPE_ridge = np.around((mean_absolute_percentage_error(df["Actual"], df["Ridge"]) * 100),2)


    result = pd.DataFrame()

    result ["StockName"] = [stock_name] * 4
    result["Model"] = mdl_list
    result["RMSE"] = [RMSE_xgb, RMSE_rf, RMSE_lasso, RMSE_ridge]
    result["MAE"] = [MAE_xgb, MAE_rf, MAE_lasso, MAE_ridge]
    result["MAPE"] = [MAPE_xgb, MAPE_rf, MAPE_lasso, MAPE_ridge]
    # result["Forecast_Bias"] = [FB_xgb, FB_rf, FB_lasso, FB_ridge]

    minpos, min_MAPE = result['MAPE'].tolist().index(min(result['MAPE'].tolist())) , min(result['MAPE'].tolist())

    model_index = {0 : "XgBoost", 1 : "Random Forest", 2 : "Lasso", 3 : "Ridge"}

    

    result.to_csv('C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Final_Stock_Forecasting\\model_results.csv', index = False)

    return (model_index[minpos], min_MAPE)




def _plot (df, stock_name, bst_model) :


# first plot with X and Y data

    df['day_of_week'] = df['Date'].dt.day_name()
    df = df[~df["day_of_week"].isin(["Saturday", "Sunday"])]

    fig, ax = plt.subplots()
    
    # Adding axes on the figure

    fig.autofmt_xdate()


    # ax[0, 1].set_title("Random Forest")
    
    # For Lasso
    ax.plot(df["Date"], df["Actual"], '-ro', label="Actual")
    ax.plot(df["Date"], df[bst_model], '-bo', label="Predicted")
    ax.legend()
    ax.set_title(bst_model)

    plt.savefig("Plot_" + stock_name +  ".png")
    df.rename(columns={bst_model : 'Best_Model'}, inplace=True)
    return df[["Date", "Actual", "Best_Model"]]

    # plt.show()
    # plt.pause(5)
    # plt.close()



if __name__ == "__main__":

    #parameters
    
    stock_name = 'AAPL'

    # Read data from s3
    client = boto3.client('s3')
    path = 's3://fortunestockdata/'+ stock_name + '/' + stock_name +  '.csv'
    df = pd.read_csv(path)

    #Calling functions
    df_pre_process = _data_preprocessing(df, stock_name)
    df_feature_engineering = _feature_engineering(df_pre_process, 14)
    df_model = _models(df_feature_engineering, 14)
    best_model, MAPE  = _accuracy (df_model, stock_name)
    predictions = _model_prediction_blind_run(df_feature_engineering,best_model,14)

    df_plot = _plot(df_model, stock_name, best_model)




    print (df.head(2))

    df.to_csv('C:\\Users\\mabbasi4\\Documents\\Courses\\5337\\Final_Stock_Forecasting\\df_plot.csv')
















