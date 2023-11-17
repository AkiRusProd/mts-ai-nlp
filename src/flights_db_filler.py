import random
from datetime import datetime, timedelta
from flights_db import FlightsDB


db = FlightsDB('flights.csv')

cities = ["Volgograd", "Saint Petersburg", "Novosibirsk", "Yekaterinburg", "Kazan", "Chelyabinsk", "Rostov", "Ufa", "Krasnoyarsk", "Perm"]


for _ in range(30):
    city = random.choice(cities)
    price = int(random.uniform(500, 1500))
    departure_date = datetime(2023, random.randint(1, 12), random.randint(1, 28), random.randint(0, 23), 0, 0)
    arrival_date = departure_date + timedelta(hours=3)
    seat_place = f"{random.choice('ABCDEF')}{random.randint(1, 30)}"
    db.add_flight(city, departure_date.strftime('%Y-%m-%d %H:%M:%S'), arrival_date.strftime('%Y-%m-%d %H:%M:%S'), seat_place, price)

print(db.get_flights())