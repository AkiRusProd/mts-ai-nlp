from typing import List, Optional, Union
import pandas as pd
import os
from dotenv import dotenv_values


env = dotenv_values(".env")
DB_PATH = env["DB_PATH"]

class FlightsDB:
    def __init__(self, filename: str = 'flights.csv'):
        self.filename: str = f"{DB_PATH}/{filename}"
        if os.path.exists(self.filename):
            self.flights: pd.DataFrame = pd.read_csv(self.filename)
        else:
            self.flights: pd.DataFrame = pd.DataFrame(columns=['city_name', 'departure_date', 'arrival_date', 'seat_place', 'price'])

    def add_flight(self, city_name: str, departure_date: str, arrival_date: str, seat_place: str, price: float) -> None:
        flight_data = pd.DataFrame({'city_name': [city_name], 'departure_date': [departure_date], 'arrival_date': [arrival_date], 'seat_place': [seat_place], 'price': [price]})
        self.flights = pd.concat([self.flights, flight_data], ignore_index=True)
        self.flights.to_csv(self.filename, index=False)

    def get_flights(self, city_name: Optional[str] = None, departure_date: Optional[str] = None) -> pd.DataFrame:
        if city_name and departure_date:
            return self.flights[(self.flights['city_name'] == city_name) & (self.flights['departure_date'] == departure_date)]
        elif city_name:
            return self.flights[self.flights['city_name'] == city_name]
        else:
            return self.flights

    def get_cities(self) -> List[str]:
        return self.flights['city_name'].unique().tolist()

    def get_departure_dates(self, city_name: str) -> List[str]:
        return self.flights[self.flights['city_name'] == city_name]['departure_date'].to_list()

    def get_arrival_date(self, city_name: str, departure_date: str) -> Optional[str]:
        flight = self.flights[(self.flights['city_name'] == city_name) & (self.flights['departure_date'] == departure_date)]
        return flight['arrival_date'].values[0] if not flight.empty else None

    def get_prices(self, city_name: str, departure_date: str) -> Optional[List[float]]:
        flight = self.flights[(self.flights['city_name'] == city_name) & (self.flights['departure_date'] == departure_date)]
        return flight['price'].values.tolist() if not flight.empty else None

    def get_ticket_ids(self, city_name: str) -> List[int]:
        return self.flights[self.flights['city_name'] == city_name].index.tolist()

    def get_ticket(self, ticket_id: int) -> Optional[pd.Series]:
        if ticket_id in self.flights.index:
            return self.flights.iloc[ticket_id]
        else:
            return None