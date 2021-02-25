import requests
from pprint import pprint
import time

# API KEY
API_key = "62b8e1f5175f87ba1db6d6968a7e3ac1"

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


# Code obtained from https://openweathermap.org/api/one-call-api

base_url = "https://api.openweathermap.org/data/2.5/onecall/timemachine?"
latitude = input("Enter latitude : ")
longitude = input("Enter longitude : ")
current_time = int(time.time()) - 24*3600 #within 5 days

Final_url = base_url + "lat=" + latitude + "&lon=" + longitude + "&dt=" + str(current_time) + "&appid=" + API_key

weather_data = requests.get(Final_url).json()

# JSON data is difficult to visualize, so you need to pretty print
pprint(weather_data)

weather_data['hourly'][12]['temp']-weather_data['hourly'][0]['temp']