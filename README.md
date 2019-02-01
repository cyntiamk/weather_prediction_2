![alt tag](https://github.com/cyntiamk/weather_prediction_2/blob/master/Resources/front_page.png?raw=true "p-values")
# Weather prediction 
Using a Machine Learning model, try to predict weather temperatures.

Project steps:
1. Aquired daily historical weather dataset from Open Weather for the city of Kyoto, Manly, Salvador, Kauai and Nice (between January 1, 2017 and December 31, 2018)
2. Extracted paramenters from JSON file, such as Temperature, Max Temperature, Min Temperature, Humidity, Atm Pressure, Wind Speed.
3. Converted all temperatures from Kelvin into Celsius, this step was necessary in order to calculate Dew Point Temperatures in the next step.
4. Calculated Dew Point Temperatures.
5. Checked for linearity of features to decide which features to utilize in the model.
6. Created features based on measurements of previous 3 days.
7. Removed all features that had a p-value greater than 0.05.
8. Tested various Regression Models and narrowed it down to 3:
  - Ridge: r2_score = 0.942 | mse = 2.98
  - Linear: r2_score = 0.95 | mse = 2.59
  - Decision Tree Regressor: r2_score = 0.95 | mse = 2.68
9. Created Python Flask, JavaScript, HTML and CSS to display predictions and current weather of 7 select cities.
10. Compared 3 remaining models and decided on Linear Regression.
11. Created Analysis page
12. Created markdown file with code snippets

### Tools/Resources
- Python: Pandas, Sklearn, Numpy
- Web: JavaScript, HTML, CSS, Flask
- Analysis: Plotly.js
- File type: JSON 
- Database: SQLite
- Source: www.openweathermap.org
