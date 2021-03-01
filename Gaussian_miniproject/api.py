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
#lat = (36, 71)
#lon = (-9, 37)

#Bounds of iceland
lat = (63, 67)
lon = (-25, -13)

# define the dictionary for GPyOpt
domain = [{'name': 'lon', 'type': 'continuous', 'domain': lon},
          {'name': 'lat', 'type': 'continuous', 'domain': lat}]


## we have to define the function we want to maximize --> validation accuracy,
## note it should take a 2D ndarray but it is ok that it assumes only one point
## in this setting
def objective_function(x):
    param = x[0]
    lon = round(param[0], 5)
    lat = round(param[1], 5)
    # create the model
    model = WindSpeed(lat, lon, current_time)

    return - model # BO want to minimize this. Thus -

# Do random assignment of initial latitude and longitude ...
np.random.seed(20)
start_size = 10
lat_rand, lon_rand = np.random.uniform(low=lat[0],high=lat[1], size=start_size), np.random.uniform(low=lon[0],high=lon[1], size=start_size)

X_init = np.array([list(x) for x in zip(lon_rand,lat_rand)])
y_init = np.array([objective_function([x]) for x in X_init]).reshape(-1,1)

# BO search:
acquisition_functions = ['MPI', 'EI', 'LCB']
exploration_values = [0.01, 0.1, 1, 2]
#exploration_values = [2, 5, 10, 20]

def BO_parameter_opt(N, X_init = X_init, y_init = y_init, domain = domain, acquisition_functions = acquisition_functions, exploration_values = exploration_values):

    cur_best_y = 0
    cur_best_ys = np.zeros(len(y_init) + N)
    cur_best_acq = None
    cur_best_exp = None
    cur_best_x = []

    for a_function in acquisition_functions:
        for i, exp_val in enumerate(exploration_values):
            opt = None
            print('-'*90)
            print("Currently running with " + a_function + " and an exploration weight of "+ str(exp_val))
            opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                                      domain=domain,# box-constrains of the problem
                                                      X=X_init, Y=y_init,
                                                      acquisition_type=a_function  # Select acquisition function MPI, EI, LCB
                                                      )

            opt.acquisition.exploration_weight=exp_val
            # See documentation of GPyOpt.models.base.BOModel() to see kernel type

            opt.run_optimization(max_iter = N)

            print("min =", np.min(opt.Y), "expval =", exp_val, "a_func = ", a_function)
            print("Location of minima: ", opt.X[np.argmin(opt.Y)])
            print("Step-index of minimum sample: ", np.argmin(opt.Y))

            if np.min(opt.Y) < cur_best_y:
                cur_best_y = np.min(opt.Y)
                cur_best_ys = opt.Y
                cur_best_acq = a_function
                cur_best_exp = exp_val
                cur_best_x = opt.X[np.argmin(opt.Y)]
                print("Updated parameters!")
            '''
            x_best = opt.X[np.argmin(opt.Y)]
            opt.plot_acquisition("modelPlot_"+a_function+"_expvalNo="+str(i), label_x="Longitude", label_y="Latitude")
            opt.plot_convergence("convergencePlot_"+a_function+"_expvalNo="+str(i))
    
            print("Coordinates with largest wind speed: latitude=" + str(x_best[1]) + ", longitude=" + str(x_best[0])
                  + "\nAcquistion function = "+a_function + ", exploration weight = " + str(exp_val))
            print("="*90)
            '''
    return cur_best_y, cur_best_ys, cur_best_x, cur_best_acq, cur_best_exp



# Random search:

def random_func(start_size, X_init = X_init, y_init = y_init, latitude = lat, longitude=lon):
    lat_rand, lon_rand = np.random.uniform(low=latitude[0],high=latitude[1], size=start_size), np.random.uniform(low=longitude[0],high=longitude[1], size=start_size)

    X_rand = np.array([list(x) for x in zip(lon_rand,lat_rand)])
    y_rand = [float(objective_function([x])) for x in X_rand]

    X_rand, y_rand = np.concatenate((X_init, X_rand)), np.concatenate(([y_init[i][0] for i in range(len(y_init))], y_rand))

    X_rand_best, y_rand_best = X_rand[np.argmin(y_rand)], np.min(y_rand)

    return X_rand, y_rand, X_rand_best, y_rand_best


# Plot comparison:
def plot_comparison(N, rand_cur_best, BO_cur_best):

    iterations = np.arange(0, N, 1)
    plt.plot(iterations, rand_cur_best, 'o-', color = 'red', label = 'Random Search')
    plt.plot(iterations, BO_cur_best, 'o-', color='blue', label='Bayesian Optimization')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Windspeed')
    plt.title('Comparison between Random Search and Bayesian Optimization')
    plt.show()
    #return BO_best_ys, rand_best_ys, BO_best_ys, x_best, cur_best_acq, cur_best_exp

N = 50
# BO
BO_best_y, BO_best_ys, x_best, cur_best_acq, cur_best_exp = BO_parameter_opt(N)

opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                                      domain=domain,
                                                      X=X_init, Y=y_init,# box-constrains of the problem
                                                      acquisition_type=cur_best_acq,  # Select acquisition function MPI, EI, LCB
                                                      )
opt.acquisition.exploration_weight=cur_best_exp
            # See documentation of GPyOpt.models.base.BOModel() to see kernel type

opt.run_optimization(max_iter = N)

# Random:
rand_best_ys = random_func(N)[1]

rand_cur_best, BO_cur_best = np.zeros(N), np.zeros(N)
for i in range(1,N+1):
    rand_cur_best[i-1] = np.min(rand_best_ys[:(i + len(y_init))]) # Tager alle de initial, og finder den current bedste over vores iterations.
    BO_cur_best[i-1] = np.min(opt.Y[:(i + len(y_init))])


plot_comparison(N, rand_cur_best, BO_cur_best)



#Plotting acquisition function, convergence:
opt.plot_acquisition("bestModelPlot_" + cur_best_acq, label_x="Longitude", label_y="Latitude")
#plt.show
opt.plot_convergence("bestConvergencePlot_" + cur_best_acq)
#plt.show

print("Coordinates with largest wind speed: latitude=" + str(x_best[1]) + ", longitude=" + str(x_best[0])
      + "\nAcquistion function = " + cur_best_acq + ", exploration weight = " + str(cur_best_exp))
print("=" * 90)
print("")
