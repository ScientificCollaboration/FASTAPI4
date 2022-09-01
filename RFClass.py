

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# Load all necessary libraries
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm().pandas()

#import torch

import sys, getopt


import numpy as np
import pandas as pd

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


# forecast monthly births with xgboost
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
#from matplotlib import pyplot
import pickle

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import OPTICS
import numpy as np; 
# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
#import matplotlib.pyplot as plt
#import seaborn as sns
#%matplotlib inline
#from statsmodels.graphics.tsaplots import plot_acf
#from statsmodels.graphics.tsaplots import plot_pacf
#plt.style.use('fivethirtyeight')

# Modelado y Forecasting
# ==============================================================================
from sklearn.linear_model import Ridge

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#from sklearn.cluster import OPTICS
import pickle
import requests
from tqdm import tqdm
tqdm.pandas()
#from pandarallel import pandarallel
#pandarallel.initialize(progress_bar=True)
from array import *
import schedule
import time
import psycopg2 as pg
import pandas.io.sql as psql

from h3 import h3
import psycopg2 as pg
import pandas.io.sql as psql
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from pydantic import BaseModel
import joblib
from fastapi import FastAPI

#app = FastAPI()

class InputDoc(BaseModel):
  text  : str

class Orders(BaseModel):
    arr: float 
    #datasetttt:float

#class Data # 2. Class which describes a single flower measurements
class Predict:
    # 6. Class constructor, loads the dataset and loads the model
    #    if exists. If not, calls the _train_model method and 
    #    saves the model
    def __init__(self):
        #self.df = datasetttt
        self.model_fname_ = 'model.pkl'
          

    def read_data(self):
        global ar
        ar = []
        global pred
        pred = []
        global datasetttt
        connection = pg.connect("host=gw-sand-toyou.net.amhub.org dbname=amdelivery_sandbox user=a.zabolotskii password=r7LJ3WSR5PAYLYV3 sslmode=require")
        dataframedelivery = psql.read_sql('SELECT * FROM delivery_order LIMIT 20000', connection)

        connection2 = pg.connect("host=gw-sand-toyou.net.amhub.org dbname=amdelivery_sandbox user=a.zabolotskii password=r7LJ3WSR5PAYLYV3 sslmode=require")
        dataframelocation = psql.read_sql('SELECT * FROM location  LIMIT 1000000', connection2)

        merged_2 = dataframedelivery.merge(dataframelocation, how='inner', left_on=["pick_up_location"], right_on=["id"])
        cols = ['lat','lon','creation_date']
        data = merged_2[cols]
        df2 = data
        df2['creation_date'] = data['creation_date'].astype('datetime64[h]')
        df2['orders'] = 1
        df2= df2.groupby(["lon","lat", "creation_date"], as_index=False)["orders"].count()
        df2 = df2.sort_values(by = ['creation_date'], ascending = [False])
        h3_level = 9
 
        def lat_lng_to_h3(row):
            return h3.geo_to_h3(
                row.lat, row.lon, h3_level)
 
        orders = df2.apply(lat_lng_to_h3, axis=1)
        df2['fid'] = orders
        df2 = df2.rename(columns={"creation_date": "time"})
        table = pd.pivot_table(df2, values='orders', index=['time'],columns=['fid'])
        table = table.fillna(0)
        table = table.sort_values(by = ['time'], ascending = [False])
   
        datasetttt = table[0:1100]
        datasetttt = datasetttt.iloc[:,:10]
        return datasetttt
    
    def fitdata(self,datasetttt):
        end_validation = 951
        end_train = 800
        for index in range(datasetttt.shape[1]):
            columnSeriesObj = datasetttt.iloc[:, index]
            columnSeriesObj = pd.Series(list(columnSeriesObj))
            forecaster = ForecasterAutoregMultiOutput(
                regressor = RandomForestRegressor(max_depth=14),
                lags = 20,
                steps = 24
                )

    
            columnSeriesObj1 = columnSeriesObj[48:1024]
            columnSeriesObj2 = columnSeriesObj[24:1000]
            i = columnSeriesObj1.index
            columnSeriesObj2.index = i
    
            param_grid = {'n_estimators': [100, 500],'max_depth': [4, 6]}

            lags_grid = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
            results_grid = grid_search_forecaster(
                        forecaster  = forecaster,
                        y           = pd.Series(list(columnSeriesObj2)),#data[:,index],table11 = pd.Series(list(table1))   
                        exog        = pd.Series(list(columnSeriesObj1)),#data[:,index],
                        param_grid  = param_grid,
                        lags_grid   = lags_grid,
                        steps       = 24,
                        metric      = 'mean_absolute_error',
                        refit       = False,
                        initial_train_size = 900,
                        return_best = True,
                        verbose     = False
                  )
            return forecaster
        
    def predictions(self):
        end_train = 800
        end_validation = 951
        for index in range(datasetttt.shape[1]):
            columnSeriesObj = datasetttt.iloc[:, index]
            columnSeriesObj = pd.Series(list(columnSeriesObj))
            columnSeriesObj1 = columnSeriesObj[48:1024]
            columnSeriesObj2 = columnSeriesObj[0:976]
            i = columnSeriesObj1.index
            columnSeriesObj2.index = i
    #columnSeriesObj3 = columnSeriesObj[24:1000]
            columnSeriesObj3 = columnSeriesObj[0:976]
        #forecaster = fitdata(datasetttt)
            global forecaster
            forecaster = ForecasterAutoregMultiOutput(
                regressor = RandomForestRegressor(max_depth=14),
                lags = 20,
                steps = 24
                )
            metric, predictions = backtesting_forecaster(
                            forecaster = forecaster,
                            y          = pd.Series(list(columnSeriesObj2)),
                            exog       = pd.Series(list(columnSeriesObj1)),
                            initial_train_size = len(columnSeriesObj[:end_validation]),
                            steps      = 24,
                            metric     = 'mean_absolute_error',
                            refit      = False,
                            verbose    = False)
            ar.append(predictions.copy())
            print(ar)
        
#def DF():        
#    arr = np.array(ar)
#    arr = arr.reshape(25,10)
#    arr = pd.DataFrame(arr,index=datasetttt[0:25].index,columns=datasetttt[0:10].columns )  
#    print(arr)           
        
        
    def save_data(self):
        forecasterd = self.fitdata(datasetttt)
        #PIK = "models.pckl"
        #pred.append(forecasterd)
        #with open("models.pckl", "wb") as f:
        #    for forecaster in pred:
        #        pickle.dump(forecasterd, f)
            
        #with open(PIK, "rb") as f:
        #    print(pickle.load(f))

        arr = np.array(ar)
        arr = arr.reshape(25,10)
        arr = pd.DataFrame(arr,index=datasetttt[0:25].index,columns=datasetttt[0:10].columns )    
        print(arr)           
            
    def main(argv):
   # if torch.cuda.is_available():
   #     dev = "cuda:0"
        
   # else:
   #     dev = "cpu"
   #     device = torch.device(dev)
    #    torch.cuda.set_device(-1)
        if __name__ == "__main__":
            
            main(sys.argv[1:])
p = Predict()             
schedule.every(24).hours.do(p.read_data)
datasetttt = p.read_data()     
schedule.every().monday.do(p.fitdata)  
p.fitdata(datasetttt)
schedule.every(24).hours.do(p.predictions) 
p.predictions()
schedule.every(24).hours.do(p.save_data)
p.save_data()