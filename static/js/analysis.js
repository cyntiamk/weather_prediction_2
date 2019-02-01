var linear = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [3.1, 2.9, 1.3, 4.0, 1.8, 2.4, 0.8],
  type: 'scatter',
  name: 'Linear'
};

var ridge = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [3.7, 3.5, 1.5, 3.6, 2.1,1.7,1.2],
  type: 'scatter',
  name: 'Ridge'
};
var DTR = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [3.3, 2.3, 1.3, 4.4,2.0,2.3,1.0],
  type: 'scatter',
  name: 'Decision Tree'
};

var data = [linear, ridge, DTR];

var layout = {
  title:'Temperature Variance Comparison (F)'
};

Plotly.newPlot("graph1", data,{responsive:true}, layout);

var linear = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [0.369, -0.635, 0.367, -1.362, -0.193, 0.411, -1.069],
  type: 'scatter',
  name: 'Linear'
};

var ridge = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [0.208, -1.758,0.128,-0.902, -0.166,0.542,-1.388],
  type: 'scatter',
  name: 'Ridge'
};
var DTR = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [0.276, -0.067, 0.052,-1.899,-0.3,0.357,-1.838],
  type: 'scatter',
  name: 'Decision Tree'
};

var data = [linear, ridge, DTR];

var layout = {
  title:'R2_SCORE on new data'
};

Plotly.newPlot("graph2", data,{responsive:true}, layout);


var linear = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [5.2, 2.76, 0.79, 7.03, 1.64, 3.20, 0.29],
  type: 'scatter',
  name: 'Linear'
};

var ridge = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [6.15, 4.77,1.08,5.53,1.60,2.74,0.33],
  type: 'scatter',
  name: 'Ridge'
};
var DTR = {
  x: ['Amsterdam', 'Irvine', 'Kauai', 'Kyoto','Nice', 'Manly','Salvador'],
  y: [5.89,1.82, 0.04,8.38,1.75, 3.60,0.38],
  type: 'scatter',
  name: 'Decision Tree'
};

var data = [linear, ridge, DTR];

var layout = {
  title:'MSE on new data'
};

Plotly.newPlot("graph3", data,{responsive:true}, layout);