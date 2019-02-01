

### Data pre-processing:
- This experiemnt was heavily based on the tutorial: https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-1/

#### Transforming data
```python
# function to convert temperatures from Kelvin to Celsius
def k_to_c (k):
    return k - 273.15
# function to calculate temperature Dew Point
def calculate_dp(T, H):
    return T - ((100 - H) / 5)

# Finding average, max and min temperatures and dew point temperature
city_df = pd.DataFrame(city_dict)
grouped_city = city_df.groupby('Date')
city_mean = grouped_city[['Avg_temp','Avg_dwp']].mean()
city_max = grouped_city[['Temp_max','Max_dwp']].max()
city_min= grouped_city[['Temp_min','Min_dwp']].min()

# Creating new dataframe with desired variables
dfs = [city_mean, city_max, city_min]
df_final = reduce(lambda left,right: pd.merge(left,right,on='Date'), dfs)
 ```
 #### Creating features
 
 ```python
 # function to create new features, adding as columns the measurement from the 3 previous days. The first 3 rows had to be padded with N with none values to keep the row consistent in lenght
 # df is the dataframe name, feature is each columns name added the number to distinguish between days and N is the number of days 
 def new_features(df, feature, N): 
    # total number of rows
    rows = df.shape[0]
    numb_days_prior_measurements = [None]*N + [df[feature][i-N] for i in range(N, rows)]
    # make a new column name of feature_N and add to DataFrame
    col_name = "{}_{}".format(feature, N)
    df[col_name] = numb_days_prior_measurements
    
features_city = ['Avg_temp', 'Avg_dwp', 'Temp_max', 'Max_dwp', 'Temp_min', 'Min_dwp']

#N is the number of days prior to the prediction, 3 days for this model
# for loop calling the function to create new features with selected variables in features_city
for feature in features_city:  
    if feature != 'Date':
        for N in range(1, 4):
            new_features(df_final, feature, N)

clean_df = df_final.dropna()
```
### Machine Learning
#### Removing noise (or high p-value features)
```python
# a p-value helps you determine the significance of your results
# what the data are telling you about the Average Temperature 
# (p-value < 0.05 strong evidence)

import statsmodels.api as sm

# separate our my predictor variables (X) from my outcome variable y
X = df[predictors]
y = df['Avg_temp']

# Add a constant to the predictor variable set to represent the Bo intercept
X = sm.add_constant(X)  

# (2) Fit the model
model = sm.OLS(y, X).fit()

# (3) evaluate the coefficients' p-values
model.summary()  

```

![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/p-value.png?raw=true "p-values")
