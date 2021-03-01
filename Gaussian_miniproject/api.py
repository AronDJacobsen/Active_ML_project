import requests
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score
import time, random, GPyOpt

# API KEY
# Remember to change key!
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
# lat = (37, 41)
# lon = (-109, -102)

#Specifying bounds of europe.
lat = (36, 71)
lon = (-9, 37)

# define the dictionary for GPyOpt
domain = [{'name': 'lon', 'type': 'continuous', 'domain': lon},
          {'name': 'lat', 'type': 'continuous', 'domain': lat}]


## we have to define the function we want to maximize --> validation accuracy,
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
def objective_function(x):
    param = x[0]
    lat = round(param[0], 5)
    lon = round(param[1], 5)
    # create the model
    model = WindSpeed(lat, lon, current_time)

    return - model # BO want to minimize this. Thus -

# Do random assignment of initial latitude and longitude ...
np.random.seed(20)
start_size = 50
lat_rand, lon_rand = np.random.uniform(low=lat[0],high=lat[1], size=start_size), np.random.uniform(low=lon[0],high=lon[1], size=start_size)

X_init = np.array([list(x) for x in zip(lat_rand,lon_rand)])
y_init = np.array([objective_function([x]) for x in X_init]).reshape(-1,1)


acquisition_functions = ['MPI', 'EI', 'LCB']
exploration_values = [0.01, 0.1, 0.5, 1]

for a_function in acquisition_functions:
    for i, exp_val in enumerate(exploration_values):
        opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                                  domain=domain,
                                                  X=X_init, Y=y_init,# box-constrains of the problem
                                                  acquisition_type=a_function,  # Select acquisition function MPI, EI, LCB
                                                  )

        opt.acquisition.exploration_weight=exp_val
        # See documentation of GPyOpt.models.base.BOModel() to see kernel type

        opt.run_optimization(max_iter = 100)

        x_best = opt.X[np.argmin(opt.Y)]

        opt.plot_acquisition("modelPlot_"+a_function+"_expvalNo="+str(i), label_x="Longitude", label_y="Latitude")
        opt.plot_convergence("convergencePlot_"+a_function+"_expvalNo="+str(i))

        print("Coordinates with largest wind speed: latitude=" + str(x_best[1]) + ", longitude=" + str(x_best[0])
              + "\nAcquistion function = "+a_function + ", exploration weight = " + str(exp_val))
        print("="*90)



"""
# Plot of guesses
lon_plot, lat_plot = zip(*opt.X)
plt.plot(lon_plot, lat_plot, "--o")
xs = np.arange(len(lon_plot))
for i, (x, y) in enumerate(zip(lon_plot, lat_plot)):
    plt.text(x, y, str(xs[i]+1), color="black", fontsize=12)
plt.margins(0.1)
"""