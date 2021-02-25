import requests
from pprint import pprint
import time
import numpy as np
import GPyOpt

# API KEY
API_key = "62b8e1f5175f87ba1db6d6968a7e3ac1"



# Code obtained from https://openweathermap.org/api/one-call-api
current_time = int(time.time()) - 24*3600 # yesterday

def WindSpeed(lat, lon, current_time):
    base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine?"
    latitude = str(lat)
    longitude = str(lon)

    Final_url = base_url + "lat=" + latitude + "&lon=" + longitude + "&dt=" + str(current_time) + "&appid=" + API_key

    weather_data = requests.get(Final_url).json()

    windspeed = weather_data['hourly'][12]['wind_speed'] #wind-speed, yesterday at 12
    return windspeed



## define the domain of the considered parameters
lon = tuple(np.arange(13, 24, 0.1, dtype=np.int))
# print(n_estimators)
lat = tuple(np.arange(63, 67, 0.1, dtype=np.int))

# define the dictionary for GPyOpt
domain = [{'name': 'lon', 'type': 'continuous', 'domain': lon},
          {'name': 'lat', 'type': 'continuous', 'domain': lat}]


## we have to define the function we want to maximize --> validation accuracy,
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
def objective_function(x):

    # create the model
    model = WindSpeed(lat, lon, current_time)

    return model


opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                          domain=domain,  # box-constrains of the problem
                                          acquisition_type='EI',  # Select acquisition function MPI, EI, LCB
                                          )

opt.acquisition.exploration_weight=0.5

opt.run_optimization(max_iter = 15)

x_best = opt.X[np.argmin(opt.Y)]

print("The best parameters obtained: longitude=" + str(x_best[0]) + ", latitude=" + str(x_best[1]))

"""
# This stores the url
base_url = "http://api.openweathermap.org/data/2.5/weather?"

# This will ask the user to enter city ID
#city_id = input("Enter a city ID : ")
city_name = input("Enter a city name : ")

# This is final url. This is concatenation of base_url, API_key and city_id
#Final_url = base_url + "appid=" + API_key + "&id=" + city_id
Final_url = base_url + "appid=" + API_key + "&q=" + city_name

# this variable contain the JSON data which the API returns
weather_data = requests.get(Final_url).json()

# JSON data is difficult to visualize, so you need to pretty print
pprint(weather_data)
"""