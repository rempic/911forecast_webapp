###  911 APIs for forcasting
import matplotlib.pyplot as plt
#import plotly
from scipy.stats import linregress
#from pandas.tools.plotting import autocorrelation_plot
#from pandas.tools.plotting import lag_plot
#from pandas.tools.plotting import bootstrap_plot
from flaskexample import app
import numpy as np
import pandas as pd
import pandasql as pdsql
from sklearn import linear_model
from sklearn import preprocessing
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import requests
import io
import pickle
import datetime 
import os

app.config['PATH_ABS_ROOT2'] = os.path.dirname(os.path.abspath(__file__))
app.config['PATH_ABS_STATIC2'] = os.path.join(app.config['PATH_ABS_ROOT2'], 'flaskexample/static')
app.config['PATH_ABS_DATAMODEL2'] = os.path.join(app.config['PATH_ABS_STATIC2'], 'model_data')


# LOAD IN REAL TIME
def load_911db_realtime():    
    url="https://storage.googleapis.com/montco-stats/tz.csv"
    d=requests.get(url).content
    d=pd.read_csv(io.StringIO(d.decode('utf-8')), header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','e'],dtype={'lat':float,'lng':float,'desc':str,'zip':str,'title':str,'timeStamp':str,'twp':str,'e':int})   
    d=pd.DataFrame(d)
    return d

def save_911db(df, dir, name):
    df.to_csv(dir + name)

# LOAD FROM LOCAL DB
def load_911db_local():
    d = [];
    d=pd.read_csv(app.config['PATH_ABS_DATAMODEL2'] + '/911db_small_20170622_19_18.csv', header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','e'],dtype={'lat':float,'lng':float,'desc':str,'zip':str,'title':str,'timeStamp':str,'twp':str,'e':int})    
    d=pd.DataFrame(d)
    return d

# type_db = 1 'EMS'
# type_db = 2 'Fire'
# type_db = 3 'Traffic'
# type_db = any number 'all'
def get_db_type(DB, type_db):
    
    if type_db == 1:
        sql1 = "select * from DB where title like 'EMS:%'"
        d3 = pdsql.sqldf(sql1, {'DB':DB})
        return pd.DataFrame(d3)
    
    if type_db == 2:
        sql1 = "select * from DB where title like 'Fire:%'"
        d3 = pdsql.sqldf(sql1, {'DB':DB})
        return pd.DataFrame(d3)
    
    if type_db == 3:
        sql1 = "select * from DB where title like 'Traffic:%'"
        d3 = pdsql.sqldf(sql1, {'DB':DB})
        return pd.DataFrame(d3)
    
    return DB


def add_data_time_columns(dt): 
    
    # ADD THE DATE AND TIME IN SINGLE COLUMNS FOR GROUPPING THE CALLS BY HOURS
    dt1 = np.zeros((dt.shape[0], 4))

    for i in range(0,dt.shape[0]):
        s = dt.timeStamp[i]
        s = s.split(' ')
        ymd = s[0]. split('-')
        hms = s[1]. split(':')
        dt1[i,0]=int(ymd[0])
        dt1[i,1]=int(ymd[1])
        dt1[i,2]=int(ymd[2])
        dt1[i,3]=int(hms[0])

    dt1 = pd.DataFrame(dt1)
    dt2 = pd.concat([dt,dt1], axis=1)

    names = dt2.columns.tolist()
    names[names.index(0)] = 'year'
    names[names.index(1)] = 'month'
    names[names.index(2)] = 'day'
    names[names.index(3)] = 'hour'
    dt2.columns = names
    dt2 = pd.DataFrame(dt2)
    
    # RE-INDEX WITH THE TIME STAMP
    sql1 = "select timeStamp, year, month, day, hour, count(*) as calls from dt2 where title like 'EMS:%' group by year, month, day, hour"
    dt3 = pdsql.sqldf(sql1, {'dt2':dt2})
    dt3 = pd.DataFrame(dt3)
    dt4 = dt3.set_index('timeStamp')
    del dt4.index.name
    
    return dt4

# this function creates the features values for the model 
# feature values <T-1>, <T-2> ... <T-n> 
    # n = num_past_events 
    # hours_single_event:
    # T-1 can be a sequence or time window wich is then averaged (<T-1>). 
    #The size of teh window is given by teh hours_single_event
    # period_part_events is the step of the sliding window
    # uncommet the DEBUG section to see the index of the time widows
# feature values:  Year, month, day, hour
# feature values:  polynomial features given from degree (if >1)
def get_feature_values(df, time_in, DEGREE, REPEATS, SHIFT,WIN_HOURS):
        
    feature_CALLS = np.zeros(REPEATS)
    feature_STATS = np.zeros(3)
    feature_DATETIME_YMDH = np.zeros(4)
    
    # SPLIT IN PAST AND FUTURE
    df_past = df[df.index<time_in]

    if df_past.shape[0]<1:
        return -1
    
    # CALLS FEATUREs VALUES (7)
    for i in range(0,REPEATS):
        j = (df_past.shape[0]-WIN_HOURS)-i
        j = j-(i*(SHIFT-1))
        feature_CALLS[REPEATS-i-1] = int(np.mean(df_past.calls[j:j+WIN_HOURS])) # calculate mean from a small interv around the point
        #DEBUG----------------------------------------------------------------
        #print(str("x:") +  str(j) + str("-") + str(j+WIN_HOURS-1))
        #-----------------------------------------------------------------
    
    # STATS (2)
    feature_STATS[0] = np.std(feature_CALLS)
    feature_STATS[1] = np.mean(feature_CALLS)
    LEN_TOT_X = (WIN_HOURS-(WIN_HOURS-SHIFT))*(REPEATS-1)+WIN_HOURS
    feature_STATS[2] =  np.abs(feature_CALLS[0]-feature_CALLS[REPEATS-1])/LEN_TOT_X

    # DATE TIME FEATUREs VALUES(4)
    feature_DATETIME_YMDH[0] = int(df_past.year[df_past.shape[0]-1])
    feature_DATETIME_YMDH[1] = int(df_past.month[df_past.shape[0]-1])
    feature_DATETIME_YMDH[2] = int(df_past.day[df_past.shape[0]-1])
    feature_DATETIME_YMDH[3] = int(df_past.hour[df_past.shape[0]-1])

    feature_VALUES = np.concatenate((feature_STATS,feature_DATETIME_YMDH,feature_CALLS,), axis=0)

    # POLYNOMIAL FEATUREs VALUES 
    if(DEGREE>1):
        degree = 3
        p = PolynomialFeatures(DEGREE,  include_bias=True)
        feature_VALUES = p.fit_transform(feature_VALUES)

    return feature_VALUES


# this takes the y (mean of y[0:num_hours]) of the test data 
# it is used only to test the prediction
def get_knownvalues_single(df, time1, WIN_HOURS):
    df_future = df[df.index>=time1]
    if df_future.shape[0]<WIN_HOURS:
        return -1
    y = np.mean(df_future.calls[0:WIN_HOURS])
    return y 

def get_prediction(features_vals, model_dir, model_type):
    sdir = model_dir + '/' + model_type + '/'
    model_ser = pickle.load(open(sdir + '911_model.sav', 'rb'))
    scaler_ser = pickle.load(open(sdir + '911_scaler.sav', 'rb'))
    features_vals_scal = scaler_ser.transform(features_vals)     
    y = model_ser.predict(features_vals_scal)
    return y

  
# add the current predicted value at the end of the data frame to get the new data frame ready for 
# fro the next prediction from the old one and so on ...
# df_in:      data frame
# calls:      the number of predicted calls 
# step_hours: the numeber of hours to add calculating the prediction time
# return the a new dataframe 
def add_new_row(df, calls, step_hours):
    time_last = df.index[df.shape[0]-1]
    time_next = pd.to_datetime(time_last) + pd.Timedelta(hours=step_hours)
    row = [time_next.year, time_next.month, time_next.day, time_next.hour, calls ]
    df.loc[str(time_next)] = row    
    return df


def get_knownvalues_multi(dt2, time0,time1, WIN_HOURS):
    
    if dt2[dt2.index>time1].shape[0]<WIN_HOURS:
        return -1 
    
    datelist = pd.date_range(time0, time1, freq='1H').tolist()
    n = len(datelist)
    
    y = np.zeros(n)

    for i in range(0,n):
        time = datelist[i].strftime("%Y-%m-%d %H:%M:%S")
        y[i] = get_knownvalues_single(dt2, time, WIN_HOURS)

    return y

def get_prediction_multi(model_type, dt2, time0,time1, DEGREE, REPEATS, STEP, WIN_HOURS):
 
    if dt2[dt2.index<time1].shape[0]<(REPEATS*WIN_HOURS):
        return -1 
    
    datelist = pd.date_range(time0, time1, freq='1H').tolist()
    n = len(datelist)
    
    y = np.zeros(n)

    for i in range(0,n):
        time = datelist[i].strftime("%Y-%m-%d %H:%M:%S")
        #print(time)
        x = get_feature_values(dt2, time, DEGREE, REPEATS,STEP,WIN_HOURS)
        y[i] = get_prediction(x, app.config['PATH_ABS_DATAMODEL2'], model_type)
       
    return y


# give the stating time and the model param returns  x_pred, y_pred, x_test, y_test with the prediction and test
# hte prediction are applied on the past and futures depending on the data and on the number of PREDICTION_HOURS_CYCLES
# in the future  
def predict_future(df_in, TIME_PRED_START, PREDICTION_HOURS_CYCLES, DEGREE, REPEATS, SHIFT, WIN_HOURS, WIN_HOURS_TRAIN ):

    y_pred = np.zeros(PREDICTION_HOURS_CYCLES)
    x_pred = ["" for x in range(PREDICTION_HOURS_CYCLES)]
    y_test = []
    x_test = []

    df = df_in.copy()
    for i in range(0,PREDICTION_HOURS_CYCLES):
       
        x = get_feature_values(df, TIME_PRED_START, DEGREE, REPEATS,SHIFT,WIN_HOURS)
        y_pred[i] = get_prediction(x, app.config['PATH_ABS_DATAMODEL2'], 'linear_regression')
        
        y = get_knownvalues_single(df_in, TIME_PRED_START, WIN_HOURS_TRAIN)
        
        TIME_PRED_START = pd.to_datetime(TIME_PRED_START) + pd.Timedelta(hours=1)
        if y >0:
            y_test = np.append(y_test, y)
            x_test = np.append(x_test, TIME_PRED_START)

        df = add_new_row(df, y_pred[i], 1)
        x_pred[i] = TIME_PRED_START
        TIME_PRED_START = TIME_PRED_START.strftime('%Y-%m-%d %H:%M:%S')

    return x_pred, y_pred, x_test, y_test




