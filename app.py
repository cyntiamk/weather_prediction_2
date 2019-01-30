import os
import pandas as pd 
import numpy as np
from datetime import datetime
from functools import reduce
import time
import csv
import sqlite3
from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)
import json
import requests

import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.externals import joblib

# Dependencies
import openweathermapy as ow

# import api_key from config file
from config import api_key

# functions to create new feataures
def new_features(df, feature, N): 
    # total number of rows
    rows = df.shape[0]
    # a list representing number of days for prior measurements of feature
    # notice that the front of the list needs to be padded with N
    # None values to maintain the constistent rows length for each N
    numb_days_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    # make a new column name of feature_N and add to DataFrame
    col_name = "{}_{}".format(feature, N)
    df[col_name] = numb_days_prior_measurements

def create_recent_features(table_name, output_table):
	# connect to sqlite database
	connex = sqlite3.connect("weather_predict.db")  
	cur = connex.cursor() 
	query = "SELECT * FROM " + table_name
	city = pd.read_sql(query, con=connex)

	city_date = []

	for day in city['Date']:
	    timestamp = datetime.strptime(day,'%Y-%m-%d %H:%M:%S')
	    day_only = datetime.strftime(timestamp,'%Y-%m-%d')
	    city_date.append(day_only)
	date = pd.DataFrame(city_date)

	city['Date'] = date.values
    
	grouped_city = city.groupby('Date')
	city_mean = grouped_city[['Mean_temp','Mean_dwp']].mean()
	city_max = grouped_city[['Max_temp','Max_dwp']].max()
	city_min= grouped_city[['Min_temp','Min_dwp']].min()

	dfs = [city_mean, city_max, city_min]

	df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
	city_renamed = df_final.rename(columns={'Mean_temp': 'Avg_temp',
											'Mean_dwp': 'Avg_dwp',
											'Max_temp': 'Temp_max',
											'Max_dwp': 'Max_dwp',
											'Min_temp':'Temp_min',                                       	
	                                       	'Min_dwp': 'Min_dwp'})

	features_city = ['Avg_temp', 'Avg_dwp', 'Temp_max', 'Max_dwp', 'Temp_min', 'Min_dwp']
	#N is the number of days prior to the prediction, 3 days for this model
	for feature in features_city:  
	    if feature != 'Date':
	        for N in range(1, 4):
	            new_features(city_renamed, feature, N)
	clean_df = city_renamed.dropna()
	clean_df.to_sql(name=output_table, con=connex, if_exists="replace", index=True)

def run_feat():
	manly_recent = 'manly_recent'
	manly_output = 'Manly_recent_features'
	create_recent_features(manly_recent,manly_output)

	nice_recent = 'nice_recent'
	nice_output = 'Nice_recent_features'
	create_recent_features(nice_recent, nice_output)

	kauai_recent = 'kauai_recent'
	kauai_output = 'Kauai_recent_features'
	create_recent_features(kauai_recent, kauai_output)

	salvador_recent = 'salvador_recent'
	salvador_output = 'Salvador_recent_features'
	create_recent_features(salvador_recent,salvador_output)

	kyoto_recent = 'kyoto_recent'
	kyoto_output = 'Kyoto_recent_features'
	create_recent_features(kyoto_recent,kyoto_output)

	amsterdam_recent = 'amsterdam_recent'
	amsterdam_output = 'Amsterdam_recent_features'
	create_recent_features(amsterdam_recent,amsterdam_output)

	irvine_recent = 'irvine_recent'
	irvine_output = 'Irvine_recent_features'
	create_recent_features(irvine_recent, irvine_output)  

def c_to_f(c):
    return ((c*9/5) + 32).round(1)

run_feat()   

app = Flask(__name__)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/weather', methods=['GET', 'POST'])
def query_owm():

	city = request.args.get('selected_city')

	data = []
	url = "http://api.openweathermap.org/data/2.5/weather?"
	units = "imperial"
	query_url = url + "appid=" + api_key + "&q=" + city +"&units="+ units

	weather_response = requests.get(query_url)
	data.append(weather_response.json())
	for measure in data:
	    current_dict = {
	        #"City": (measure['name']),
	        "Description": (measure['weather'][0]['main']),
	        "Temperature": (measure['main']['temp']),
	        "Humidity": (measure['main']['humidity']),
	        "Wind_speed": (measure['wind']['speed'])
	    }
	
	current_df = pd.DataFrame(current_dict, index=[0])
	
	return current_df.to_json(orient='records')


@app.route('/prediction', methods=['GET','POST'])
def predicting_temp():
	connex = sqlite3.connect("weather_predict.db") 
	cur = connex.cursor() 

	city = request.args.get('selected_city')

	query_city = city + '_recent_features'
	query = "SELECT * FROM " + query_city

	city_df = pd.read_sql(query, con=connex).set_index('Date')
	sorted_city = city_df.sort_values('Date', ascending=False)
	
	predictors = ['Avg_temp_1', 'Avg_temp_2', 'Avg_temp_3', 'Temp_max_1', 'Temp_max_2',
       'Temp_min_1', 'Temp_min_3', 'Avg_dwp_2', 'Avg_dwp_3', 'Min_dwp_1',
       'Min_dwp_3']

	X = sorted_city[predictors]
	y= sorted_city['Avg_temp']

	scaler = joblib.load(open('concat_scaler.save', 'rb'))
	X_scaled = scaler.transform(X)

	# import ridge model to predict temperature with recent features created
	model = pickle.load(open('ridge_concat_feats.pkl', 'rb'))

	y_predicted = model.predict(X_scaled)


	# apply function to convert temperature into Fahrenheit
	actual_f = c_to_f(sorted_city['Avg_temp'])

	avg_f = c_to_f(y_predicted)

	avg_fahrenheit = []

	for i in range(0, len(avg_f)):
	    avg_fahrenheit.append(int(avg_f.item(i)))

	# grab values and add to dataframe  
	sorted_city['Actual_avg_temp'] = ''
	sorted_city['Predicted_temp']= ''

	sorted_city['Actual_avg_temp'] = round(actual_f).astype(int)
	sorted_city['Predicted_temp'] = avg_fahrenheit

	# Create new dataframe with only Average temperature collected and compare with predicted temperature 
	predictions_df = sorted_city[['Actual_avg_temp','Predicted_temp']]
	predictions_df['Difference'] =  predictions_df.Actual_avg_temp - predictions_df.Predicted_temp

	# save prediction dataframe into csv
	predictions_df.to_sql(city + '_prediction',con=connex, if_exists="replace", index=True)
	return predictions_df.iloc[0].to_json(orient='records')

if __name__ == "__main__":
	app.run(debug=True)
