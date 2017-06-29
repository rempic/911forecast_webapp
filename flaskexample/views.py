from flask import render_template
from flaskexample import app
import pandas as pd
from flask import request
import numpy as np
import base64
from io import BytesIO
import urllib.parse
import matplotlib.dates as md
import datetime
from datetime import timedelta
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import pytz

import insightmodel as mymodel

app.config['PATH_ABS_ROOT'] = os.path.dirname(os.path.abspath(__file__))
app.config['PATH_ABS_STATIC'] = os.path.join(app.config['PATH_ABS_ROOT'], 'static')
app.config['PATH_ABS_DATAMODEL'] = os.path.join(app.config['PATH_ABS_STATIC'], 'model_data')




#@app.route('/')
#@app.route('/index')
#def index():
#    return render_template("index.html")




@app.route('/')
@app.route('/index')
def first_pediction():
    return update_predictions_main(False) # set up this at True to load the local web site first 

#@app.route('/')
#@app.route('/index')
@app.route('/update', methods=['GET','POST'])
def update_predictions():
    return update_predictions_main(False)



def update_predictions_main(bLoadLocalDB):
    
    # -----------------------
    # LOAD DATA
    # -----------------------
    df_all = []
    if bLoadLocalDB == False:
        df_all = mymodel.load_911db_realtime()
    
    if bLoadLocalDB == True:
        df_all = mymodel.load_911db_local()
    

    df_ems = mymodel.get_db_type(df_all, 1)
    df_model = mymodel.add_data_time_columns(df_ems)
    n = df_model.shape[0]

    # -----------------------
    # RUN MODEL
    # -----------------------
    DEGREE = 3  #polynomial 
    SHIFT = 24  # period
    WIN_HOURS = 4   #window x
    WIN_HOURS_TRAIN = 2 # window y
    REPEATS = 4 #number of features 
    PREDICTION_HOURS_CYCLES = 51  # total number of predictions including past and future (depends on the statrting time)

    time_start = ""
    time_now = "" 
    time_show = ""
    time_yesterday = ""

    if bLoadLocalDB == True :
        time_start = '2017-06-22 00:00:00'
        time_show = '2017 06 18'
    else:
        time_now_temp = datetime.datetime.now(pytz.timezone("America/New_York"))
        time_show = time_now_temp.strftime("%b %d %Y")

        time_next_1hour = time_now_temp + timedelta(hours=1)
        time_next_5hour = time_now_temp + timedelta(hours=5)
        time_next_10hour = time_now_temp + timedelta(hours=10)
        time_next_1hour = time_next_1hour.strftime("%H:00")
        time_next_5hour = time_next_5hour.strftime("%H:00")
        time_next_10hour = time_next_10hour.strftime("%H:00")

        time_now = time_now_temp.strftime("%Y-%m-%d 00:00:00")
        
        time_yesterday = time_now_temp - timedelta(hours=24)
        time_yesterday = time_yesterday.strftime("%Y-%m-%d %H:00:00")
        time_start = time_yesterday
 
    x_pred,y_pred,x_test,y_test = mymodel.predict_future(df_model, time_start, PREDICTION_HOURS_CYCLES, DEGREE,REPEATS, SHIFT, WIN_HOURS,WIN_HOURS_TRAIN )

      # ----------------------------
    # SHOW  PLOT (1DAY, from today)
    # ----------------------------
    # PLOT THE GRAPH
    marker_size = 100
    font_size = 100
    font_size_pred = 150
    tick_spacing = 0.05

    shift = 2
    n_test = len(x_test)
    n_pred= len(x_pred)
    pre_err = 3

    fig, ax = plt.subplots(figsize=(130,60));

    # PAST 
    plt.plot(x_test, y_test,'-b', markersize=marker_size, lw=4)
    plt.plot(x_test, y_test,'.b', markersize=marker_size)
    plt.grid('off')

    # PLOT THE ERROR for the prediction 
    err_old_pred = np.zeros(n_test) + pre_err 
    err_new_pred = np.zeros(n_pred-n_test) + pre_err   

    x_pred_s = x_pred[0:n_pred-shift]
    y_pred_s = y_pred[shift:n_pred]

    plt.plot(x_pred_s,y_pred_s,'.--',  color = 'red',  markersize=marker_size, lw=8) 
    plt.fill_between(x_pred_s, y_pred_s-pre_err, y_pred_s+pre_err, facecolor='k', alpha=0.1)

    # PRED NEW
    plt.plot(x_pred[n_test:n_pred-shift], y_pred[n_test+2:n_pred],'.--r', markersize=marker_size*2, lw=8)

    # PLOT SETTINGS
    ax.set_xticks(x_pred)

    # format axis
    xfmt = md.DateFormatter('%H:00')
    ax.xaxis.set_major_formatter(xfmt)

  
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # legend
    red_patch = mpatches.Patch(color='red', label='predicted 911 calls')
    blue_patch = mpatches.Patch(color='blue', label='past 911 calls')
    plt.legend(handles=[red_patch, blue_patch], fontsize=font_size)

    # ticks and axis-labels
    #plt.xlabel("today:" + time_show + " 911 calls forecast", fontsize=font_size)
    plt.ylabel('Number of 911 calls (Montgomery County, PA)', fontsize=font_size+5)

    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off', labelsize=font_size)
    plt.tick_params(axis='x',  labelsize=font_size)
    plt.xticks(rotation=70)

    plt.text(x_pred[np.int(n_test*1.5)], 4, "Forecast on " + time_show, color='red', fontsize=font_size_pred)
    plt.text(x_pred[np.int(n_test*1.5)], 3, time_next_1hour + ": " + str(np.int(y_pred[n_test+shift])) + " calls", color='red', fontsize=font_size_pred)
    plt.text(x_pred[np.int(n_test*1.5)], 2, time_next_5hour + ": " + str(np.int(np.sum(y_pred[n_test+shift:n_test+shift+5]))) + " calls", color='red', fontsize=font_size_pred)
    plt.text(x_pred[np.int(n_test*1.5)], 1, time_next_10hour + ": " + str(np.int(np.sum(y_pred[n_test+shift:n_test+shift+10]))) + " calls", color='red', fontsize=font_size_pred)
    
    k = 0
    for tick in ax.xaxis.get_ticklabels():
        if k>=n_test-2:
            tick.set_color('red')
            tick.set_weight('bold')
        k = k + 1
        

    plt.show()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    #plot_url = base64.b64encode(img.getvalue())
    plot_prediction_url = urllib.parse.quote(base64.b64encode(img.read()).decode()) 


    # ----------------------------
    # SHOW  PLOT (1DAY, from today)
    # ----------------------------
    f, ax=plt.subplots(1,1)
    ax.set_xticks(x_pred)
    xfmt = md.DateFormatter('%H:00')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks( rotation=90 )

    red_patch = mpatches.Patch(color='red', label='predicted calls')
    blue_patch = mpatches.Patch(color='blue', label='past calls')

    n_test = len(x_test)
    n_pred= len(x_pred)

    plt.plot(x_pred[0:n_test], y_pred[0:n_test],'.', color = '0.75', markersize=100)
    plt.plot(x_pred[n_test:n_pred], y_pred[n_test:n_pred],'.r', markersize=100)
    plt.plot(x_pred[n_test:n_pred], y_pred[n_test:n_pred],'.w', markersize=20)

    
    plt.plot(x_test, y_test,'-b', markersize=40)
    plt.plot(x_test, y_test,'.b', markersize=20)


    blue_patch = mpatches.Patch(color='blue', label='past calls')
    plt.xlabel('day', fontsize=7)
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off', labelsize=10)
    plt.tick_params(axis='x',  labelsize=10)

    plt.legend(handles=[red_patch, blue_patch], fontsize=10)

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    #plot_url = base64.b64encode(img.getvalue())
    plot_model_url = urllib.parse.quote(base64.b64encode(img.read()).decode()) 

    # -----------------------
    # Build MAP
    # -----------------------
    df_locations_ltln = df_ems[df_ems.timeStamp > time_now]
    df_locations_ltln = df_locations_ltln.ix[:,0:2] # take the columns need
    df_locations_ltln.columns=['latitude','longitude']

    main = {}
    coord = np.ndarray.tolist(np.array(df_locations_ltln))
    cc = [{"coordinates":i} for i in coord]
    main["features"] = [{"geometry":i} for i in cc]
    json_main = json.dumps(main)
    json_locations = 'eqfeed_callback(' + json.dumps(main) + ')'
    
    json_locations = json_locations.replace(' ', '')
    
    #o = os.path.join(app.config['UPLOAD_FOLDER'], 'map_loc.js')
    sDir = app.config['PATH_ABS_DATAMODEL'] 
    #sDir = 'static/model_data'
    print(app.config['PATH_ABS_DATAMODEL'])
    file_name = sDir + '/map_loc.js'
    with open(file_name, 'w') as file:
        file.write(json_locations)
        file.close()
    
    
    c1 = np.array(df_locations_ltln.ix[:,0])
    c2 = np.array(df_locations_ltln.ix[:,1])
    sC1 = ''
    sC2 = ''
    for i in range(0, len(c2)) :
        sC1 = sC1 + " " + str(c1[i])
        sC2 = sC2 + " " + str(c2[i])

    return render_template('index.html', plot_url=plot_prediction_url, coord1=sC1, coord2=sC2, time_now=time_show)



