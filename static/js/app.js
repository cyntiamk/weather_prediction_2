

var selected_city = d3.select('#dropdownSelect').node().selectedOptions[0].value;


function get_owm(selected_city) {

	var selected_city = d3.select('#dropdownSelect').node().selectedOptions[0].value;
	//console.log(selected_city)
	
	d3.json(`/weather?selected_city=${selected_city}`).then((data) => {
		//console.log(data[0].City)
		var panel = d3.select("#panel-status");
		panel.html("");
		var weatherInfo = 
					{City: data[0].City,
					Description: data[0].Description,
					Temperature: Number((data[0].Temperature).toFixed(0)),
					Humidity: data[0].Humidity ,
					Wind: data[0].Wind_speed};
		//console.log(weatherInfo)
		Object.entries(weatherInfo).forEach(([key,value]) => {
			panel.append("h6").text(`${key}: ${value}`);
		})
	});
	d3.json(`/prediction?selected_city=${selected_city}`).then((predData) => {
	console.log(predData[1]) 
	
	var predictedTemp = {Predicted_temp: predData[1]};
	console.log(predictedTemp)
	Object.entries(predictedTemp).forEach(([key,value]) =>{
		var span = document.getElementById("prediction").innerHTML =`${value}`;
		span.html("")
	})
	});

	var bgImage = {
		Amsterdam: 'url(/static/css/img/ams.jpg)',
		Irvine: 'url(/static/css/img/irvine.jpg)',
		Lihue: 'url(/static/css/img/kauai.jpg)',
		Kyoto: 'url(/static/css/img/kyoto.jpg)',
		Nice: 'url(/static/css/img/nice.jpg)',
		Manly: 'url(/static/css/img/manly.jpg)',
		Salvador: 'url(/static/css/img/salvador.jpg)'
		};
		var selected_value = bgImage[selected_city]
		console.log(selected_value)

		var bgBody = document.body.style["background-image"] = selected_value;


}

get_owm(selected_city)

