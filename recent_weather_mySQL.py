# Dependencies
import csv
import openweathermapy as ow
import pandas as pd
import requests
import pprint
import time
from datetime import datetime
import os
import sqlite3

# import api_key from config file
from config import api_key


def grab_json(city_name, city_sql_name): 
    # Opens file if exists, else creates file
    connex = sqlite3.connect("weather_predict.db") 
    # This object lets us actually send messages to our DB and receive results
    cur = connex.cursor()   
    
    data = []

    url = "http://api.openweathermap.org/data/2.5/weather?"
    units = "metric"
    city = url + "appid=" + api_key + "&q=" + city_name +"&units="+ units

    weather_response = requests.get(city)
    data.append(weather_response.json())

    date_obj = []
    temp = []
    max_temp = []
    min_temp = []
    humidity = []
    pressure = []
    wind_speed = []
    clouds = []
    description = []

    for measure in data:
        date_obj.append(measure['dt'])
        temp.append(measure['main']['temp'])
        max_temp.append(measure['main']['temp_max'])
        min_temp.append(measure['main']['temp_min'])
        pressure.append(measure['main']['pressure'])
        humidity.append(measure['main']['humidity'])
        wind_speed.append(measure['wind']['speed'])
        clouds.append(measure['clouds']['all'])
        description.append(measure['weather'][0]['main'])

    def calculate_dp(T, H):
        return T - ((100 - H) / 5)

    dew_point = []
    for T ,H in zip(temp, humidity):
        dp = calculate_dp(T,H)
        dew_point.append(dp)

    max_dew = []
    for T ,H in zip(max_temp, humidity):
        dp = calculate_dp(T,H)
        max_dew.append(dp)

    min_dew = []
    for T ,H in zip(min_temp, humidity):
        dp = calculate_dp(T,H)
        min_dew.append(dp)

    date = []
    for seconds in date_obj:
        timestamp = datetime.utcfromtimestamp(seconds)
        day = datetime.strftime(timestamp,'%Y-%m-%d %H:%M:%S')
        date.append(day) 

    city_weather = {
        "Date": date,
        "Mean_temp": temp,
        "Max_temp": max_temp,
        "Min_temp": min_temp,
        "Mean_dwp": dew_point,
        "Max_dwp": max_dew,
        "Min_dwp": min_dew,
        "Pressure": pressure,
        "Humidity": humidity,
        "Wind": wind_speed,
        "Clouds": clouds,
        "Description": description
    }

    df = pd.DataFrame(city_weather)
    
    # save into mySQL database
    import sqlalchemy
    database_username = 'root'
    database_password = 'Pufferin@1'
    database_ip       = 'localhost'
    database_name     = 'weather_prediction'
    database_connection = sqlalchemy.create_engine('mysql+mysqlconnector://{0}:{1}@{2}/{3}'.
                                                   format(database_username, database_password, 
                                                          database_ip, database_name))
    df.to_sql(con=database_connection, name=city_sql_name+'_recent', if_exists='append',index=False, index_label='Date')
    
def run_all_json():
    ams_name = 'Amsterdam'
    ams_sql_name = 'amsterdam'
    grab_json(ams_name, ams_sql_name)
    
    kyo_name = 'Kyoto'
    kyo_sql_name = 'kyoto'
    grab_json(kyo_name, kyo_sql_name)
    
    nic_name = 'Nice'
    nic_sql_name = 'nice'
    grab_json(nic_name, nic_sql_name)
    
    kau_name = 'Lihue'
    kau_sql_name = 'kauai'
    grab_json(kau_name, kau_sql_name)
    
    sal_name = 'Salvador'
    sal_sql_name = 'salvador'
    grab_json(sal_name, sal_sql_name)
    
    man_name = 'Manly'
    man_sql_name = 'manly'
    grab_json(man_name, man_sql_name)

    irv_name = 'Irvine'
    irv_sql_name = 'irvine'
    grab_json(irv_name, irv_sql_name)

while(True):
    run_all_json()
    time.sleep(3600)