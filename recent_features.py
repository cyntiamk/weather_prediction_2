import pandas as pd
import numpy as np
from datetime import datetime
from functools import reduce
import time

# function to create new features based on 3 previous days
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

def create_recent_features(csv_path, city_initial, city_name):
    city = pd.read_csv(csv_path)
    city_date = []

    for day in city['Date']:
        timestamp = datetime.strptime(day,'%Y-%m-%d %H:%M:%S')
        day_only = datetime.strftime(timestamp,'%Y-%m-%d')
        city_date.append(day_only)
    date = pd.DataFrame(city_date)

    city['Date'] = date.values
    
    del city['Unnamed: 0']
    
    grouped_city = city.groupby('Date')
    city_mean = grouped_city[['Mean_temp','Mean_dwp']].mean()
    city_max = grouped_city[['Max_temp','Max_dwp']].max()
    city_min= grouped_city[['Min_temp','Min_dwp']].min()

    dfs = [city_mean, city_max, city_min]

    df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
    city_organized = df_final[['Mean_temp','Max_temp','Min_temp','Mean_dwp','Max_dwp','Min_dwp']]
    city_renamed = city_organized.rename(columns={'Mean_temp': city_initial+'_temp','Max_temp': city_initial+'_max','Min_temp':city_initial+'_min',
                                           'Mean_dwp': city_initial+'_dwp','Max_dwp': city_initial+'_mx_dwp','Min_dwp': city_initial+'_mi_dwp'})
    features_city = list(city_renamed.columns.values)
    #N is the number of days prior to the prediction, 3 days for this model
    for feature in features_city:  
        if feature != 'Date':
            for N in range(1, 4):
                new_features(city_renamed, feature, N)
    city_renamed.to_csv(city_name +'_recent_features.csv')
    
def run_all():
    manly_path = 'manly_recent.csv'
    man_initial = 'Man'
    man_name = "manly"
    create_recent_features(manly_path, man_initial, man_name)

    nice_path = 'nice_recent.csv'
    nic_initial = 'Nice'
    nic_name = "nice"
    create_recent_features(nice_path, nic_initial, nic_name)

    kauai_path = 'kauai_recent.csv'
    kauai_initial = 'Kau'
    kauai_name = "kauai"
    create_recent_features(kauai_path, kauai_initial, kauai_name)

    sal_path = 'salvador_recent.csv'
    sal_initial = 'Sal'
    sal_name = "salvador"
    create_recent_features(sal_path, sal_initial, sal_name)

    kyo_path = 'kyoto_recent.csv'
    kyo_initial = 'Kyo'
    kyo_name = "kyoto"
    create_recent_features(kyo_path, kyo_initial, kyo_name)
    
    ams_path = 'amsterdam_recent.csv'
    ams_initial = 'Ams'
    ams_name = "amsterdam"
    create_recent_features(ams_path, ams_initial, ams_name)

    irv_path = 'irvine_recent.csv'
    ams_initial = 'Irv'
    ams_name = "irvine"
    create_recent_features(ams_path, ams_initial, ams_name)


    
while(True):
    run_all()
    time.sleep(86400)
