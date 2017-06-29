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
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    return update_predictions_main(True)

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
    DEGREE = 3
    SHIFT = 24
    WIN_HOURS = 4
    WIN_HOURS_TRAIN = 2
    REPEATS = 4
    PREDICTION_HOURS_CYCLES = 48   

    time_now = "" 
    time_now_show = ""

    if bLoadLocalDB == True :
        time_now = '2017-06-22 00:00:00'
        time_now_show = '2017 06 18'
    else:
        time_now_temp = datetime.datetime.now(pytz.timezone("America/New_York"))
        time_now = time_now_temp.strftime("%Y-%m-%d 00:00:00")
        time_now_show = time_now_temp.strftime("%Y %m %d")
    #time_now = datetime.datetime.now()
    #time_now = time_now.strftime("%Y-%m-%d 00:00:00")
    #time_now_show = time_now.strftime("%Y . %m . %d")
    x_pred,y_pred,x_test,y_test = mymodel.predict_future(df_model, time_now, PREDICTION_HOURS_CYCLES, DEGREE,REPEATS, SHIFT, WIN_HOURS,WIN_HOURS_TRAIN )

    

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

    #plt.plot(x_pred[0:n_test], y_pred[0:n_test],'.', color = '0.75', markersize=260)
    #plt.plot(x_pred[n_test:n_pred], y_pred[n_test:n_pred],'.r', markersize=260)
    #plt.plot(x_pred[n_test:n_pred], y_pred[n_test:n_pred],'.w', markersize=40)
    #plt.plot(x_pred[(n_test):n_pred], y_pred[(n_test):n_pred],'-w', markersize=20)
    plt.plot(x_test, y_test,'-b', markersize=40)
    plt.plot(x_test, y_test,'.b', markersize=20)


    blue_patch = mpatches.Patch(color='blue', label='past calls')

    #plt.xlabel('Time (hour)', fontsize=20)
    plt.xlabel('day', fontsize=7)
    #plt.ylabel('Calls', fontsize=20)
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

    return render_template('index.html', plot_url=plot_model_url, coord1=sC1, coord2=sC2, time_now=time_now_show)



