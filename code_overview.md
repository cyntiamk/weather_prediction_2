

### Data pre-processing:
- This link: https://stackabuse.com/using-machine-learning-to-predict-the-weather-part-1/ was very helpful for this experiment.

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
 ![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/df_initial.png?raw=true "df_final")
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

 ![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/feats_nan.png?raw=true "df_final")

![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/df_feats.png?raw=true "features")
### Machine Learning
#### Removing noise (or high p-value features)
```python
# a p-value helps you determine the significance of your results
# what the data are telling you about the Average Temperature 
# (p-value < 0.05 strong evidence)
```
![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/predictors.png?raw=true "df_final")

```python

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

```python
X = X.drop('Max_dwp_1', axis=1)

# (5) Fit the model 
model = sm.OLS(y, X).fit()

# After all p-values greater than 0.05 have been removed, drop the 'const' column and use the rest as you X

X_clean = X.drop('const', axis=1)
```
![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/feats_list.png?raw=true "feats_list")

#### Scaling the X and fitting in the models
```python
from sklearn.preprocessing import StandardScaler

# Create a StandardScater model and fit it to the training data
X_scaler = StandardScaler().fit(X_clean)

# Save the scaler to be used on the new data
from sklearn.externals import joblib
scaler_filename = "final_scaler.save"
joblib.dump(X_scaler, scaler_filename)

from sklearn.linear_model import LinearRegression
# define the model
linear = LinearRegression()

# train the model/ fit the model 
linear.fit(X_scaled, y)

# predict
y_linear_prediction = linear.predict(X_scaled)

# use evaluation metrics to determine model performance
```
![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/model_eval.png?raw=true "model_eval")
```python
#save model
import pickle
with open('linear_final_model.pkl', 'wb') as file:
    pickle.dump(linear, file)
```

### New data collection
#### Open Weather API
```python
# function with simple API query to gather new data for prediction
url = "http://api.openweathermap.org/data/2.5/weather?"
units = "metric"
city = url + "appid=" + api_key + "&q=" + city_name +"&units="+ units

weather_response = requests.get(city)
data.append(weather_response.json())

# timer running every 60 minutes gathering date for the 7 cites
```python

while(True):
    run_all_json()
    time.sleep(3600)
# Store into SQLite
```
#### Flask
1. Query OWM for current weather conditions on selected city
2. Retrieve recent weather stored in SQLite 
3. Create features for selected city
4. Scale X with saved scaler
5. Predict the Average Temperature 
6. Convert into Fahrenheit with the function
```python
def c_to_f(c):
    return ((c*9/5) + 32).round(1)
```
7. Submit to JavaScript and save to SQLite
8. Display result into HTML
9. All the steps are repeated every time a new city is selected.

#### JavaScript
```javascript
d3.json(`/prediction?selected_city=${selected_city}`).then((predData) => {
	//console.log(predData[1]) 
	var predictedTemp = {Predicted_temp: predData[1]};
	//console.log(predictedTemp)
	Object.entries(predictedTemp).forEach(([key,value]) =>{
		var span = document.getElementById("prediction").innerHTML =`${value}`;
		span.html("")
})
});
```
